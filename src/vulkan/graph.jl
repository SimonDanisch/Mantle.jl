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

Device() = Device(VulkanAPI())


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
    win::RenderWindow
end

Window(width::Integer, height::Integer; title::AbstractString = "", vsync::Bool = false) =
    LavaWindow(RenderWindow(width, height; title = String(title), vsync))

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

"""
Usage bits an element type asks for beyond the ordinary ones, which is one type
and one bit: a buffer of draw commands is read by the command processor, and a
buffer without `INDIRECT_BUFFER_BIT` is a validation error at the draw rather
than where it was allocated.

This is `bufferusage`'s answer for this backend — "what memory can host a
`T`" is Vulkan vocabulary, so the backend owns it.
"""
extrausage(::Type) = UInt32(0)
extrausage(::Type{DrawIndirectCommand}) = UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT)
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
passed(d::LavaDevice, f) = query_timeline(d.bq) >= f

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
"""
The swapchain image is externally indexed: acquire returns whatever the
presentation engine chooses, not `frame % n`. So it is its own type, and `run!`
brackets the acquire, the wait and the present rather than any user code.
"""
struct LavaSurface <: Resource
    win::Any
end

Surface(g, win) = (s = LavaSurface(win); push!(g.surfaces, s); s)
Surface(g, w::LavaWindow) = Surface(g, w.win)

# A render pass targets either the window or an offscreen framebuffer. These four
# are the only places that difference shows.
target_view(s::LavaSurface) = s.win.views[s.win.current_image_idx + 1]
target_image(s::LavaSurface) = s.win.images[s.win.current_image_idx + 1]
target_extent(s::LavaSurface) = s.win.extent
target_format(s::LavaSurface) = s.win.format

target_view(fb::LavaFramebuffer) = fb.color_view
target_image(fb::LavaFramebuffer) = fb.color_image
target_extent(fb::LavaFramebuffer) = VK.Extent2D(fb.width, fb.height)
target_format(fb::LavaFramebuffer) = fb.color_format

# What state the target is in when the pass starts. The window arrives from
# acquire; an offscreen target was last read by the copy that took it to the
# window. Both are discarded by a clearing pass, so this only matters when the
# pass loads. A placed target has no contents at all until something writes it,
# and after aliasing it is back in that state, so it starts from Undefined.
initial_usage(::LavaSurface) = Present
initial_usage(::LavaFramebuffer) = CopySrc

# What state a resource is in when a replay begins. `nothing` means the first use
# establishes it and no transition into it is needed, which is the answer for
# every buffer: buffers have no layout, so there is nothing to transition from.
initial_state(::Any) = nothing
initial_state(s::LavaSurface) = initial_usage(s)
initial_state(fb::LavaFramebuffer) = initial_usage(fb)
# Not `nothing`: an image whose first use is a colour attachment still needs the
# layout transition out of UNDEFINED, and `nothing` would emit none.

# ── graph ─────────────────────────────────────────────────────────────────────
struct DrawCall
    shader::Any
    args::Tuple
    count::Any
    frag_args::Tuple
end

# `draw!` does not take fragment arguments yet — Lava's `draw!` has taken a
# `frag_args` tuple for a while and this is the field it will arrive in. Empty
# is what the pipeline phase compiles against today.
DrawCall(shader, args, count) = DrawCall(shader, args, count, ())

mutable struct Pass
    name::String
    kind::Symbol
    targets::Vector{Any}                # render: the colour attachments. copy: the source.
    loads::Vector{LoadOp}        # render only, one per colour attachment
    depth::Any                          # render only, and only if one was given
    depth_load::Union{Nothing,LoadOp}
    dst::Any                            # copy only
    draws::Vector{DrawCall}
    usages::Vector{Pair{Int,Type}}
    dispatches::Vector{Any}
end

Pass(name, kind) = Pass(String(name), kind, Any[], LoadOp[], nothing, nothing, nothing,
                        DrawCall[], Pair{Int,Type}[], Any[])

"""The colour attachment a pass configures itself from: extent and viewport are
the same for all of them, so the first answers for the set."""
# The attachment a pass takes its render area from. A depth-only pass — which is
# what a shadow map is — has no colour target, and then the depth one is it.
first_target(p::Pass) = isempty(p.targets) ? p.depth : first(p.targets)

# The load op, lowered. `Discard` is the one that needs saying: it is the only
# way to reach DONT_CARE, and a pass that covers every pixel should not pay to
# load what it is about to overwrite.
loadop(::KeepOp) = VK.ATTACHMENT_LOAD_OP_LOAD
loadop(::DiscardOp) = VK.ATTACHMENT_LOAD_OP_DONT_CARE
loadop(::Clear) = VK.ATTACHMENT_LOAD_OP_CLEAR

clearvalue(::LoadOp) = nothing
clearvalue(c::Clear) = NTuple{4,Float32}(c.value)

"""What a depth attachment clears to. One number, not four, and `nothing` loads."""
depthclear(::LoadOp) = nothing
depthclear(c::Clear) = Float32(c.value)

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
abstract type TransientResource <: Resource end

mutable struct TransientBuffer{T} <: TransientResource
    n::Int
    first::Int
    last::Int
    # Where in the pool this landed. The BLOCK is kept, not just the fused
    # address, because a Vulkan buffer barrier scopes to (VkBuffer, offset, size)
    # and an address alone cannot name the buffer.
    block::Any            # ::BufferBlock once placed
    offset::Int
end

Base.length(t::TransientBuffer) = t.n
Base.eltype(::TransientBuffer{T}) where {T} = T
"""What to call a transient in an error, since it has no name of its own."""
describe(t::TransientBuffer{T}) where {T} = "Transient.Buffer($T, $(t.n))"
count(t::TransientBuffer) = t.n
stride(::TransientBuffer) = 1
nbytes(t::TransientBuffer{T}) where {T} = t.n * sizeof(T)
alignment(::TransientBuffer) = 256

"""
A transient render target.

The `VK.Image` exists from the moment the handle does, because its memory
requirements are what the placer needs and only a real image can be asked. It
costs no memory until bound, and the graph is built once, so this is not a
per-frame cost.
"""
mutable struct TransientImage{T} <: TransientResource
    width::Int
    height::Int
    format::VK.Format
    usage::VK.ImageUsageFlag
    image::VK.Image
    req::NamedTuple{(:size, :alignment, :type_bits),Tuple{Int,Int,UInt32}}
    first::Int
    last::Int
    view::Any
    memory::Any
    # What this target takes its size from, or nothing for a fixed size. A window
    # changes size under a plan, and a depth buffer that does not change with it
    # stops covering the render area.
    source::Any
end

Base.size(t::TransientImage) = (t.width, t.height)
Base.eltype(::TransientImage{T}) where {T} = T
describe(t::TransientImage{T}) where {T} = "Transient.Image($T, ($(t.width), $(t.height)))"
nbytes(t::TransientImage) = t.req.size
alignment(t::TransientImage) = t.req.alignment

"""
Which allocation a transient belongs in.

Buffers and images are placed separately rather than in one arena. Vulkan permits
mixing them, but only with `bufferImageGranularity` padding between a linear and
an optimal-tiled resource, and getting that wrong is aliasing corruption that no
validation layer reports. Two arenas cost one extra allocation and make the rule
unnecessary.
"""
struct Buffers end
struct Images end
arena(::TransientBuffer) = Buffers()
arena(::TransientImage) = Images()

target_view(t::TransientImage) = t.view
target_image(t::TransientImage) = t.image
aspect(::TransientImage{T}) where {T} = aspect(T)
target_extent(t::TransientImage) = VK.Extent2D(t.width, t.height)
target_format(t::TransientImage) = t.format
initial_usage(::TransientImage) = Undefined
initial_state(t::TransientImage) = initial_usage(t)

"""
Buffers by byte size, so renaming one is a recycle rather than an allocation.

Renaming — write a fresh buffer and swap it in, rather than overwrite the one the
GPU is reading — only pays off if getting the fresh buffer is cheap. It is,
because the sizes repeat exactly: the same resource updated every frame asks for
the same size every time, so the free list hits from the second update onward.

`retire!` does not free. The GPU may still be reading the outgoing buffer for as
long as the frame that bound it is in flight, so it goes back on the free list
only once that frame has been waited on, which is what `recycle!` is called after.
"""
struct Recycler
    free::Dict{Int,Vector{Any}}
    retiring::Vector{Tuple{Int,Any,UInt64}}
end

Recycler() = Recycler(Dict{Int,Vector{Any}}(), Tuple{Int,Any,UInt64}[])

"""Take a host-visible buffer of `n` bytes, recycled if one is available.

Keyed negatively so host and device buffers of the same size never share a free
list — handing a device-local buffer to a memcpy would be a segfault, not a
wrong picture."""
function takehost!(r::Recycler, bq, n::Integer)
    pool = get(r.free, -Int(n), nothing)
    if pool !== nothing && !isempty(pool)
        return pop!(pool)
    end
    host_buffer(bq, n)
end

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

"""Hand a buffer back, reusable once the queue timeline passes `signal`."""
retire!(r::Recycler, store, nbytes::Integer, signal::Integer) =
    push!(r.retiring, (Int(nbytes), store, UInt64(signal)))

"""
Move everything the GPU has finished with onto the free lists.

Gated on the timeline rather than on a frame count, because how many frames are
in flight is not this code's business and changing it must not turn recycling
into a use-after-free.
"""
function recycle!(r::Recycler, bq)
    now = query_timeline(bq)
    keep = 0
    for (bytes, store, signal) in r.retiring
        if signal <= now
            push!(get!(() -> Any[], r.free, bytes), store)
        else
            keep += 1
            r.retiring[keep] = (bytes, store, signal)
        end
    end
    resize!(r.retiring, keep)
    r
end

mutable struct LavaGraph <: Graph
    dev::LavaDevice
    passes::Vector{Pass}
    surfaces::Vector{LavaSurface}
    transients::Vector{TransientResource}
    transient_by_id::Dict{Int,TransientResource}
    ids::IdTable          # both directions; see `IdTable`
    updates::Vector{Any}
    recycler::Recycler
    # Interning for `use(...; range = ...)`. `ids` is an IdDict, so two `use`
    # calls naming the same slice would otherwise be two objects and two ids —
    # and a resource that is not the same resource in two passes has no hazards
    # between them, which is the one answer that must not be reachable by
    # accident. Keyed by value, so the same slice is the same sub-resource.
    views::Dict{Tuple{Int,UnitRange{Int}},Any}
end

Graph(dev::LavaDevice) =
    LavaGraph(dev, Pass[], LavaSurface[], TransientResource[],
              Dict{Int,TransientResource}(), IdTable(), Any[], Recycler(),
              Dict{Tuple{Int,UnitRange{Int}},Any}())

function Transient.Buffer(g::LavaGraph, ::Type{T}, n::Integer) where {T}
    t = TransientBuffer{T}(Int(n), typemax(Int), 0, nothing, 0)
    push!(g.transients, t)
    t
end

"""
What an image of this element type is for, in the two places Vulkan asks: the
usage flags it is created with, and the aspect its view and its barriers name.

`Float32` is a depth attachment, because `vkformat` makes it `D32_SFLOAT` and
nothing else in Vulkan is a single-component 32-bit float attachment. Both
answers come from the element type so they cannot disagree — a depth image with a
colour view is a validation error at the first draw, and one created without
`DEPTH_STENCIL_ATTACHMENT_BIT` fails at creation.
"""
imageusage(::Type) = COLOR_USAGE
# The same three as COLOR_USAGE, with the attachment bit that matches the aspect:
# a depth target is worth copying out (a test that asserts the depth buffer beats
# one that asserts its effect on colour) and worth sampling (depth of field,
# ambient occlusion, anything that reads the z it just wrote).
imageusage(::Type{Float32}) = VK.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              VK.IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK.IMAGE_USAGE_SAMPLED_BIT

aspect(::Type) = VK.IMAGE_ASPECT_COLOR_BIT
aspect(::Type{Float32}) = VK.IMAGE_ASPECT_DEPTH_BIT

# The same question asked of a resource. A swapchain image and a framebuffer are
# colour by construction; only a transient carries an element type to ask.
aspect(::Any) = VK.IMAGE_ASPECT_COLOR_BIT

"""Which attachment slot a target fills, asked of the target and not of the
argument position, so the two orders of `screen => Clear(c), z => Clear(1f0)`
mean the same thing."""
isdepth(x) = aspect(x) == VK.IMAGE_ASPECT_DEPTH_BIT

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
    props = VK.get_physical_device_format_properties(ctx.physical_device, fmt)
    have = props.optimal_tiling_features
    VK = Vulkan
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

"""
    Transient.Image(graph, T, (width, height); srgb = false, usage = imageusage(T))
    Transient.Image(graph, T, target)

A render target the compiler places, so two targets whose lifetimes do not
overlap share bytes.

`T` is the pixel type and is the whole format: `RGBA{Float16}` is
`R16G16B16A16_SFLOAT`. Readback of one gives a `Matrix{T}` and a kernel writing
one writes `T`, which an enum would not have given.

Given a *target* rather than a size, it follows that target: a depth buffer for a
window is `Transient.Image(g, Float32, screen)`, and it is resized with the
window. A fixed size is the right answer only when it genuinely is fixed, because
an attachment that stops covering the render area is undefined rendering.
"""
function Transient.Image(g::LavaGraph, ::Type{T}, size::Tuple{Integer,Integer};
                                srgb::Bool = false, source = nothing,
                                usage::VK.ImageUsageFlag = imageusage(T)) where {T}
    ctx = g.dev.ctx
    width, height = size
    fmt = vkformat(VulkanAPI(), T; srgb)
    checkusage(ctx, fmt, usage)
    img = image_2d(ctx, width, height, fmt, usage)
    t = TransientImage{T}(Int(width), Int(height), fmt, usage, img,
                          image_requirements(ctx, img),
                          typemax(Int), 0, nothing, nothing, source)
    push!(g.transients, t)
    t
end

Transient.Image(g::LavaGraph, ::Type{T}, source; kw...) where {T} =
    Transient.Image(g, T, extent_size(source); source, kw...)

extent_size(x) = (e = target_extent(x); (Int(e.width), Int(e.height)))

"""
Give a tracking transient the size its source now has, and say whether it moved.

The image is recreated rather than resized, because a `VkImage` has its extent
from creation. That invalidates the placement — a different size needs different
offsets — which is why the caller recompiles rather than patching the old plan.
"""
function refit!(t::TransientImage{T}) where {T}
    t.source === nothing && return false
    w, h = extent_size(t.source)
    (w, h) == (t.width, t.height) && return false
    ctx = vk_context()
    t.width, t.height = w, h
    t.image = image_2d(ctx, w, h, t.format, t.usage)
    t.req = image_requirements(ctx, t.image)
    t.view = nothing
    t.memory = nothing
    return true
end

refit!(::TransientResource) = false

"""
Record that this pass touches `t`.

The interval recorded here is in declaration order and is provisional: Liveness
recomputes it from the scheduled order, because reordering passes is exactly what
changes a transient's lifetime.
"""
function touch!(g::LavaGraph, t::TransientResource)
    i = length(g.passes)
    t.first = min(t.first, i)
    t.last = max(t.last, i)
    g.transient_by_id[resourceid(g, t)] = t
    t
end
touch!(g::LavaGraph, x) = (resourceid(g, x); x)

# `resourceid(g::LavaGraph, r)` is NOT defined here. `resourceid(g::Graph, r)`
# (runtime/dispatch.jl) is the same line, `LavaGraph <: Graph`, and Mantle
# exports it — so this was a SECOND function of the same name in this module,
# agreeing with core's by coincidence rather than by construction.

"""
What kind of resource an id names, so barrier tracking starts in the right state
machine. Images have layouts and buffers do not, and a buffer tracked as an image
would emit layout transitions for a resource that has none.
"""
resourcekind(::Any) = BufferKind()
resourcekind(::TransientImage) = ImageKind()
resourcekind(::LavaSurface) = ImageKind()
resourcekind(::LavaFramebuffer) = ImageKind()

"""
A render pass. `target => clear` clears; a bare target loads what is there.

The target is a usage like any other, recorded before the body runs so it is
first in the sequence. Deriving it from `initial_usage` instead would be a second
statement of the same thing, and would be wrong the moment two passes render to
one target: the second would transition from the target's declared initial state
rather than from the colour attachment the first left it in.

A depth target is another attachment, spelled the same way, and which slot an
attachment fills comes from the target rather than from its position or a keyword:

    z = Transient.Image(g, Float32, (w, h))
    render!(g, "scene", screen => Clear(bg), z => Clear(1f0)) do p

`z` is a depth attachment because a `Float32` image is `D32_SFLOAT` and nothing
else in Vulkan is a single-component 32-bit float attachment. Its barrier and its
layout are then derived from the `Depth` usage like any other.

Several colour attachments are the same list again, in order, and the fragment
shader writes them by returning a tuple — element `i` goes to attachment `i`:

    render!(g, "gbuffer", albedo => Clear(bg), normals => Discard, z => Clear(1f0)) do p
        draw!(p, GBUF, bind(p, mesh, mvp), mesh.positions)   # fragment returns (c, n)
    end

A pass with only a depth target is a pass all the same, and it is what a shadow
map is: nothing is shaded, and the render area comes from the depth attachment
because there is nothing else to take it from. Its fragment shader returns
`nothing`, which is how a shader says it writes no attachment.

    render!(g, "shadow", shadowmap => Clear(1f0)) do p
        draw!(p, DEPTHONLY, args, drawcmd)
    end
"""
function render!(f, g::LavaGraph, name::AbstractString, attachments...)
    isempty(attachments) && throw(ArgumentError("a render pass needs a target"))
    p = Pass(name, :render)
    push!(g.passes, p)
    for a in attachments
        tgt = a isa Pair ? first(a) : a
        load = a isa Pair ? last(a) : Keep
        load isa LoadOp ||
            throw(ArgumentError("a render target takes Clear(value), Keep or Discard, got $load"))
        touch!(g, tgt)
        if isdepth(tgt)
            p.depth === nothing ||
                throw(ArgumentError("a pass takes one depth target, and this one was given two"))
            p.depth, p.depth_load = tgt, load
            # Tested and written, and no stencil aspect: `NoAccess` is what says
            # the image has none, which is what picks the combined layout over the
            # separate-aspect ones.
            push!(p.usages, resourceid(g, tgt) => Depth{ReadWrite,NoAccess,discards(load)})
        else
            push!(p.targets, tgt)
            push!(p.loads, load)
            push!(p.usages, resourceid(g, tgt) =>
                  (discards(load) ? ColorAttachment{true} : ColorAttachment{false}))
        end
    end
    f(PassHandle(g, p))
    p
end

struct PassHandle
    graph::LavaGraph
    pass::Pass
end

"""
The binding a pass gets from an attribute. A `Buffer` and a `Scalar` of the same
element type both erase to `Attr{T}`, differing only in `stride`, which is a
field rather than a type parameter.

That is what makes a scalar and a per-element attribute one pipeline: the binding
tuple is the pipeline key, so if the two spellings produced different types they
would produce different shaders.
"""
struct Attr{T,R} <: Resource
    resource::R
    stride::Int32
end

stride(a::Attr) = a.stride
count(a::Attr) = count(a.resource)
Base.length(a::Attr) = length(a.resource)

"""Constructing a usage is how a pass gets a binding. There is no path from a
resource to a draw that skips it."""
function Attribute(p::PassHandle, r)
    push!(p.pass.usages, resourceid(p.graph, r) => Vertices)
    Attr{eltype(r),typeof(r)}(r, Int32(stride(r)))
end

"""A draw count that lives on the device, wrapped so recording dispatches on it."""
struct Commands{R}
    resource::R
end

"""
What a draw takes its vertex count from.

A number is one. A resource is its length, so a draw over a buffer of points
covers exactly the points there are. A buffer of `DrawIndirectCommand` is the
third answer and the only one the host never learns: the count is read by the
command processor from device memory, so a compute pass in the same frame can
decide it. The element type is what says which, because a buffer of draw commands
is not something anything else would be.
"""
drawover(p::PassHandle, n) = n
drawover(p::PassHandle, n::Buffer{DrawIndirectCommand}) = indirectcount!(p, n)
drawover(p::PassHandle, n::TransientBuffer{DrawIndirectCommand}) = indirectcount!(p, n)

function indirectcount!(p::PassHandle, n)
    push!(p.pass.usages, resourceid(p.graph, n) => Indirect)
    touch!(p.graph, n)
    Commands(n)
end

function draw!(p::PassHandle, shader, args, n; frag_args = ())
    # Here rather than at compile: the pipeline has one push constant range, so a
    # draw with arguments on both stages is a mistake in the call, and by the time
    # a shader is compiled it surfaces as one stage failing to take an argument it
    # never declared.
    isempty(args) || isempty(frag_args) || throw(ArgumentError(
        "draw!: arguments were given to both stages, and a pipeline has one push " *
        "constant range. Put them on the stage that reads them and pass what the " *
        "other needs as a varying."))
    push!(p.pass.draws, DrawCall(shader, args, drawover(p, n), frag_args))
end

"""
A slice of a buffer, as its own resource.

Two passes writing disjoint halves of one buffer do not race, and tracking the
buffer whole says they do — a barrier between them orders memory neither touches.
Handing the slice its own id is what lets the existing per-resource walk answer
that without knowing anything about ranges: disjoint slices are disjoint
resources, and the hazard set falls out unchanged.

`range` is in elements, and the barrier is scoped to exactly those bytes. The
kernel still receives the whole buffer — a range declares what a pass *touches*,
not what it can address.
"""
struct BufferRange
    parent::Any
    range::UnitRange{Int}
end

storage(v::BufferRange) = storage(v.parent)
resourcekind(::BufferRange) = BufferKind()

"""
Byte span of a usage, for the barrier that scopes to it.

`pool_offset + offset`, not `offset`: a `VkBufferMemoryBarrier2`'s range is
relative to the **VkBuffer**, and Lava suballocates, so a resource's own offset
is into its `MemoryBlock` rather than into the buffer the barrier names. This is
the same sum `inplace!` and `rename!` compute for a copy, and for the same
reason.

Without it every scoped barrier reads `offset = 0` on a buffer whose resources
begin tens of megabytes in — a memory dependency over a range nothing in the
pass touches. Nothing broke, because desktop drivers treat the range as advisory
and flush at the stages given; it is still a barrier that does not describe the
hazard it was derived for, and sync validation is entitled to say so.
"""
barrierspan(r, st) = (UInt64(st.buf[].pool_offset + st.offset),
                      UInt64(sizeof(eltype(st)) * prod(st.dims)))
# A pooled transient's storage is a `LavaDeviceArray` — `(ptr, dims)`, which
# names no buffer and carries no offset. The transient knows both, and its
# `offset` is already relative to the block's buffer, so no pool_offset here.
barrierspan(t::TransientBuffer, st) = (UInt64(t.offset), UInt64(nbytes(t)))
"""The `VkBuffer` a barrier names. Same split as `barrierspan`."""
barrierbuffer(r, st) = st.buf[].buffer
barrierbuffer(t::TransientBuffer, st) = t.block.buffer

barrierspan(v::BufferRange, st) =
    (UInt64(st.buf[].pool_offset + st.offset + (first(v.range) - 1) * sizeof(eltype(st))),
     UInt64(length(v.range) * sizeof(eltype(st))))

function slice(g::LavaGraph, x, range::UnitRange{Int})
    # `length`, not `length(storage(x))`: a transient has no storage until the
    # placer gives it some, and a range is declared while the graph is built.
    n = length(x)
    (first(range) >= 1 && last(range) <= n) || throw(ArgumentError(
        "use(): range $range is outside the buffer's 1:$n."))
    pid = resourceid(g, x)
    get!(g.views, (pid, range)) do
        BufferRange(x, range)
    end
end

"""
    use(pass, x; read, write, range = nothing, unordered = false)

The ordinary case: this pass reads or writes this resource. Named usages survive
only where the role can be picked wrongly.

`range` narrows the claim to a slice, in elements. Two passes that name disjoint
slices of one buffer get no barrier between them, and one that does name a slice
gets a barrier scoped to exactly those bytes.

`unordered` says the order of this access against another unordered one does not
change the result — commutative atomics, or writes to disjoint elements. Two
passes that BOTH say it get no barrier between them however much they overlap;
anything else still does, so forgetting it anywhere gives the barrier back rather
than producing a race. A wavefront tracer's per-pixel radiance is the case it
exists for: half a dozen stages do nothing to it but `atomic +=`, and ordering
them against each other serialises passes whose queues are disjoint.
"""
function use(p::PassHandle, x; read::Bool = false, write::Bool = false,
                    range::Union{Nothing,UnitRange{Int}} = nothing,
                    unordered::Bool = false)
    read || write || throw(ArgumentError("use() needs read, write, or both"))
    S = Storage{BufferKind, Access{read, write}}
    U = unordered ? Unordered{S} : S
    r = range === nothing ? x : slice(p.graph, x, range)
    push!(p.pass.usages, resourceid(p.graph, r) => U)
    # The parent is what the kernel gets, and what liveness has to see touched.
    touch!(p.graph, x)
end

# Whether a dispatch was given a workgroup size is a type, not a branch: the
# launch is per pass per frame and this keeps the call site one expression.
kernelfor(k, ::Nothing) = k(LavaBackend())
kernelfor(k, group) = k(LavaBackend(), group)

# The body goes in `dispatches` beside the `Dispatch`es rather than in a field of
# its own: the compile walks that vector and this is one more thing it can find
# there, so `Pass` does not grow a field only one kind ever sets.
# The four hooks core's `custom!`/`compute!` are written against.
newpass(::LavaGraph, name::AbstractString, kind) = Pass(name, kind)
handle(g::LavaGraph, p::Pass) = PassHandle(g, p)
dispatches(p::Pass) = p.dispatches
passes(g::LavaGraph) = g.passes


"""The single pass every `Update` shares, so one pair of barriers covers them all."""
function updates_pass!(g::LavaGraph)
    for p in g.passes
        p.kind === :update && return p
    end
    p = Pass("updates", :update)
    pushfirst!(g.passes, p)
    p
end

function Update(g::LavaGraph, buf; range = nothing)
    p = updates_pass!(g)
    touch!(g, buf)
    return registerupdate!(g.updates, p.usages, resourceid(g, buf), buf, range)
end

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
write_update!(g::LavaGraph, bq, r::UpdateRef, data::Buffer) =
    write_update!(g, bq, r, storage(data))

# A scalar attribute is a one-element buffer, so a new value is a one-element
# write: in place, inline in the command buffer, four to sixteen bytes riding
# along with the frame. Renaming would be absurd for that, and the old
# `update!(scalar, x)` route flushes the queue to make its write safe, which is a
# stall per changed colour.
write_update!(g::LavaGraph, bq, r::UpdateRef, x) =
    (inplace!(bq, r.resource, [x], 1); nothing)

function write_update!(g::LavaGraph, bq, r::UpdateRef, data::AbstractVector{T}) where {T}
    dst = r.resource
    n = length(data) * sizeof(T)
    n == 0 && return
    if r.range === nothing && length(data) == length(dst)
        rename!(g, bq, dst, data)
    else
        inplace!(bq, dst, data, r.range === nothing ? 1 : first(r.range))
    end
    nothing
end

"""
Where the bytes already are, which is a third thing the call can mean.

Data that is already in device memory — produced by a kernel, a broadcast, or
anything else that never went through the host — needs no staging buffer and no
`memcpy`: the copy is device to device and the host never sees the array. This is
the same two routes as above, differing only in where the source is, so it is a
method rather than a branch.
"""
function rename!(g::LavaGraph, bq, dst::Buffer{T,1}, data::LavaArray{T,1}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)
    fview = deviceview(dst.dev, fresh)

    smb, fmb = data.buf[], fview.buf[]
    cmd_copy_buffer!(bq, smb, fmb, nbytes;
                          src_off = smb.pool_offset + data.offset,
                          dst_off = fmb.pool_offset + fview.offset)

    dst.store = fresh
    retire!(g.recycler, old, dst.capacity * sizeof(T),
            ensure_active_batch!(bq).signal_value)
    nothing
end

function inplace!(bq, dst, data::LavaArray{T,1}, from::Integer) where {T}
    store = storage(dst)
    dmb, smb = store.buf[], data.buf[]
    cmd_copy_buffer!(bq, smb, dmb, length(data) * sizeof(T);
                          src_off = smb.pool_offset + data.offset,
                          dst_off = dmb.pool_offset + store.offset + (Int(from) - 1) * sizeof(T))
    nothing
end

# A Mantle buffer says the same thing as its store, so it takes the same route.
rename!(g::LavaGraph, bq, dst::Buffer{T,1}, data::Buffer{T,1}) where {T} =
    rename!(g, bq, dst, storage(data))
inplace!(bq, dst, data::Buffer, from::Integer) =
    inplace!(bq, dst, storage(data), from)

"""In-place, inline in the command buffer. Falls back to renaming when the
update is too big for `cmd_update_buffer` to carry."""
function inplace!(bq, dst, data::AbstractVector{T}, from::Integer) where {T}
    store = storage(dst)
    mb = store.buf[]
    off = mb.pool_offset + store.offset + (Int(from) - 1) * sizeof(T)
    n = length(data) * sizeof(T)
    if n <= 65536 && n % 4 == 0 && off % 4 == 0
        src = data isa Vector{T} ? data : collect(data)
        GC.@preserve src VK.cmd_update_buffer(
            ensure_active_batch!(bq).cmd_buf, mb.buffer,
            UInt64(off), UInt64(n), Ptr{Cvoid}(pointer(src)))
    else
        upload!(store, data)      # partial and large: rare, and it stalls
    end
    nothing
end

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
function rename!(g::LavaGraph, bq, dst::Buffer{T,1}, data::AbstractVector{T}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)
    host = takehost!(g.recycler, bq, nbytes)

    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src Base.unsafe_copyto!(host.mapped_ptr, Ptr{UInt8}(pointer(src)), nbytes)

    fview = deviceview(dst.dev, fresh)
    fmb = fview.buf[]
    cmd_copy_buffer!(bq, host.buffer, fmb, nbytes;
                          dst_off = fmb.pool_offset + fview.offset)

    dst.store = fresh
    signal = ensure_active_batch!(bq).signal_value
    retire!(g.recycler, old, dst.capacity * sizeof(T), signal)
    retire!(g.recycler, host, -nbytes, signal)
    nothing
end

function copy!(g::LavaGraph, name::AbstractString, dst, src)
    p = Pass(name, :copy)
    push!(p.targets, src)
    p.dst = dst
    push!(g.passes, p)
    touch!(g, src); touch!(g, dst)
    push!(p.usages, resourceid(g, src) => CopySrc)
    push!(p.usages, resourceid(g, dst) => CopyDst)
    p
end

# ── plan ──────────────────────────────────────────────────────────────────────
struct CompiledDraw{A<:Tuple,C}
    # Concrete throughout: `Any` here made every field access in the per-draw
    # record path a dynamic lookup, which is where the frame's allocations were.
    compiled::CompiledGraphicsPipeline
    shader::Lava.LavaGfxShader
    args::A
    count::C
    # Where this draw's arguments live inside a slot of the plan's argument
    # memory. Fixed at compile time, because the set of draws and the size of
    # each one's arguments are.
    argoff::Int
    argsize::Int
end

"""
One dispatch, resolved the way a draw is: the shader is compiled when the plan
is, and a frame only writes arguments and records.

Going through `KernelAbstractions` per frame instead looked equivalent and was
not. Its launch plan is keyed on `Base.get_world_counter()`, so defining a method
anywhere — every eval in a session — misses the cache, and the miss is an
unconditional recompile that shells out to `spirv-opt`. That is ~20 ms, and
because waiting on a subprocess is a task switch it happens *between* the
swapchain acquire and the present. The draws never had this: they hold their
pipeline. Now the dispatches do too, and the only thing that compiles a kernel is
building a plan.

`kernel` is the kernel function after `Adapt`, which for a `@kernel` is a
singleton — checked when the plan is built, because a closure over device arrays
would be resolved once here and then go stale the first time an argument is
renamed.
"""
struct CompiledDispatch{K,A<:Tuple,I,N,R,O}
    # Concrete throughout, for the reason `CompiledDraw` is: an `Any` field turns
    # every access in the per-dispatch record path into a dynamic lookup.
    launch::LaunchPlan
    iter::I                             # IterPlan for `nd0`
    nd0::N                              # the ndrange the iteration plan was built for
    obj::O                              # the KA kernel, for an ndrange that moves
    kernel::K
    args::A
    ndrange::R
    tlas::Bool                          # whether the pipeline was built for ray query
    argoff::Int
    argsize::Int
end

"""
Argument memory owned by the plan, in slots — one per frame that can be in
flight.

The plan knows its draws and their argument sizes at compile time, so it can lay
them out once and write only values into them per frame. That removes the whole
question the argument pool exists to answer: nothing is allocated per draw, so
nothing has to work out when it may be reused. The slot is what the GPU is still
reading, and a slot is reused only after the timeline passes the frame that used
it — the same rule as everything else here, and the only rule.

`signal[i]` is the timeline value the frame using slot `i` will signal. Waiting on
it before writing is what makes "K slots" correct rather than hopeful; with K
larger than the frames the queue keeps in flight, that wait never blocks.
"""
mutable struct ArgMemory
    store::LavaArray{UInt8,1}
    address::UInt64
    ptr::Ptr{UInt8}
    stride::Int
    signal::Vector{UInt64}
    slot::Int
end

const ARG_SLOTS = 3
argalign(n::Integer) = (Int(n) + 255) & ~255

"""Lay out every draw in the plan, and take the memory once."""
function ArgMemory(dev::LavaDevice, passes::AbstractVector)   # of PassPlan, defined below
    stride = 0
    for pp in passes
        for d in pp.draws
            stride += argalign(d.argsize)
        end
        for d in pp.dispatches
            stride += argalign(d.argsize)
        end
    end
    stride = max(argalign(stride), 256)
    store = LavaArray{UInt8,1}(undef, (stride * ARG_SLOTS,); bq = dev.bq, unified = true)
    mb = store.buf[]
    ArgMemory(store, mb.address, Ptr{UInt8}(mb.mapped_ptr), stride,
              zeros(UInt64, ARG_SLOTS), 0)
end

"""Take the next slot, once the GPU is done with what it holds."""
function nextslot!(am::ArgMemory, bq)
    am.slot = mod1(am.slot + 1, ARG_SLOTS)
    want = am.signal[am.slot]
    want == 0 && return am.slot
    # The frame that last used this slot may never have been handed to the queue.
    # A plan with a surface submits every frame in `present_frame!`, but one
    # without a surface submits when something asks it to — so `ARG_SLOTS` frames
    # of a headless plan all sit in the same recording batch, and the wait below
    # is then for a value the queue has not been given and never will be. It
    # blocks in `vkWaitSemaphores`, which is a foreign call, so the process stops
    # dead with no stack. Needing the slot back is precisely the reason to submit.
    want > bq.next_timeline && submit!(bq)
    if query_timeline(bq) < want
        wait_semaphores!(bq, VK.SemaphoreWaitInfo([bq.timeline_sem], [want]))
    end
    am.slot
end

slotbase(am::ArgMemory) = (am.slot - 1) * am.stride

# ── lowering a transition into a Vulkan barrier ───────────────────────────────
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

function emit_barrier!(bq, b::ImageBarrier)
    barrier = VK._ImageMemoryBarrier2(
        b.old, b.new,
        VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
        target_image(b.resource),
        VK._ImageSubresourceRange(aspect(b.resource),
                                           UInt32(0), UInt32(1), UInt32(0), UInt32(1));
        src_stage_mask = b.src_stage, src_access_mask = b.src_access,
        dst_stage_mask = b.dst_stage, dst_access_mask = b.dst_access)
    batch = bq.active_batch
    batch === nothing && (batch = ensure_active_batch!(bq))
    VK._cmd_pipeline_barrier_2(batch.cmd_buf,
        VK._DependencyInfo([], [], [barrier]))
    nothing
end

"""
One pass, its compiled draws, and the barriers that have to run before it.

Both are derived. The swapchain image arrives in whatever state acquire left it
and the pass needs it as a colour attachment, so that transition falls out of the
usage sequence rather than being written by hand.
"""
struct PassPlan
    pass::Pass
    draws::Vector{CompiledDraw}
    dispatches::Vector{CompiledDispatch}
    images::Vector{ImageBarrier}        # layout changes, one barrier each
    pre::Vector{Transition}      # what this pass needs before it runs
    barrier::Any                        # one memory barrier covering the rest
    # Whether any of this pass's dispatches sizes itself on the device. Decided
    # at compile because it decides how the pass is recorded, and asking a
    # `Vector{CompiledDispatch}` per frame is a dynamic call per element.
    indirect::Bool
end

PassPlan(pass, draws, dispatches, images, pre, barrier) =
    PassPlan(pass, draws, dispatches, images, pre, barrier,
             any(d -> d.ndrange isa DeviceRange, dispatches))

"""
Two timestamps per pass and a ring of samples, or nothing at all.

A profiler is a plan-level thing rather than a device-level one because a pass is:
Lava's own timing is per dispatch and keyed by kernel name, which cannot see a
render pass, cannot see an update, and cannot tell two dispatches of one kernel
apart. Slots are `2i-1, 2i` for pass `i` in scheduled order.

`pending` says a frame's timestamps have been written and not yet read. They are
read without `WAIT_BIT`: a frame still in flight is skipped rather than waited
for, so turning profiling on does not change the frame time it reports.
"""
mutable struct Profiler
    pool::VK.QueryPool
    period_ns::Float64
    nslots::Int
    names::Vector{String}
    host_ns::Vector{Vector{Float64}}    # one ring per pass
    gpu_ns::Vector{Vector{Float64}}
    pending::Bool
end

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

"""Keep the last `NSAMPLES`, so a plan that runs for an hour does not grow."""
function sample!(ring::Vector{Float64}, x::Real)
    push!(ring, x)
    length(ring) > NSAMPLES && popfirst!(ring)
    ring
end

# Mutable because a window resize re-places every transient, and what comes back
# is a different set of slabs and different barriers. The plan is the user's
# handle, so it is updated rather than replaced.
mutable struct LavaPlan <: Plan
    graph::LavaGraph
    transitions::Vector{Transition}
    passes::Vector{PassPlan}
    pipelines::Set{Any}
    slabs::Vector{Any}                  # the SHARED region of each arena, not owned
    arenas::Vector{Any}                 # …and which arena each one is, for remap!/free!
    offsets::Vector{Int}                # per transient, relative to its region
    peak::Int
    naive::Int
    profiler::Union{Nothing,Profiler}
    alias::Bool                         # kept, so a recompile means what the first one did
    coalesce::Bool
    policy::Policy
    args::ArgMemory                     # laid out at compile, written per frame
    # The recording `bake!` took, or `nothing` while the plan records per run.
    # It pins the argument slot it was captured in — `slotbase` is folded into
    # every address the command buffer holds — so a baked plan stops rotating
    # slots and `nextslot!` is not called for it.
    baked::Any
end

"""The backend object behind a resource: what actually gets bound or copied."""
storage(a::Attr) = storage(a.resource)
# What a kernel receives: `LavaDeviceArray` is `(ptr, dims)` and owns nothing, so
# there is no ownership question to answer here — which is why Mantle holding the
# array type removes the problem instead of managing it.
# A borrowed view, not an owning array: the `DataRef` releaser is a no-op, so
# `copy` here bumps a refcount that frees nothing and Mantle stays the owner.
# `todevice` turns this into the `(ptr, dims)` a kernel receives; a host-side
# caller (copy_framebuffer!, a library) gets the handle it needs.
storage(t::TransientBuffer{T}) where {T} =
    LavaArray{T,1}(copy(t.block.ref), (t.n,); offset = t.offset)

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
adaptor(bq) = LavaAdaptor(ensure_active_batch!(bq))

# In two steps, and the order matters for one thing: an acceleration structure.
# `rawargs` is what the caller gave, with `Ref`s read and Mantle's own resources
# resolved to their storage; `devargs` is that after Lava's conversion. A ray
# query needs the `HWTLAS` bound as a descriptor, and adapting an
# `HWAdaptedAccel` deliberately strips it — the device side of a ray query is a
# variable, not a pointer the kernel carries. So the TLAS is looked for in the
# RAW arguments, which is where it still exists. Asking the adapted ones finds
# nothing, and a shading kernel then compiles with ray query disabled and fails
# in the emitter rather than at the call site.
rawargs(args::Tuple) = map(argvalue, args)
devargs(ad, raw::Tuple) = map(a -> Adapt.adapt(ad, a), raw)

"""The resource a usage ultimately names: a slice and a vertex binding both
forward their storage to a parent, so neither has an identity of its own to test."""
rootresource(x) = x
rootresource(v::BufferRange) = rootresource(v.parent)
rootresource(a::Attr) = rootresource(a.resource)

"""
Whether an `Update` on this resource can move it.

Only the whole-buffer route renames: `write_update!` renames when the ref has no
range *and* the data is the buffer's whole length, and writes in place otherwise.
So an `Update(g, buf; range = 1:100)` never moves its target and keeps a scoped
barrier; a bare `Update(g, buf)` may, and gives it up.

Through a slice as well as directly: `slice(g, buf, 1:100)` is a different object
from `buf`, so an identity test against the update refs misses it — and a slice of
a renamed buffer is exactly as stale as the buffer, having no storage of its own.

Asked of the graph rather than of the resource because the resource cannot know:
a `LavaBuffer` is the same type either way, and whether it is renameable is a
property of how the graph was declared.
"""
renameable(g::LavaGraph, r) =
    any(u -> u.resource === rootresource(r) && u.range === nothing, g.updates)

"""
The barrier a pass needs, as one memory barrier with stages and access ORed over
everything it waits for. Built at compile time: a plan is compiled once and
replayed, so constructing VK.jl wrapper objects per frame would be ~2.3 kB of
allocation per barrier for a value that never changes.

The low-level `_` form specifically. VK.jl's wrapper converts to the C struct
on every call and allocates 1840 bytes doing it; `_DependencyInfo` through
`_cmd_pipeline_barrier_2` allocates nothing. Measured, not assumed.

`nothing` when the pass waits for nothing, which is the point: two passes over
disjoint resources then have no barrier between them at all.
"""
function build_pass_barrier(g, ts::Vector{Transition})
    isempty(ts) && return nothing
    be = VulkanAPI()
    mem = VK._MemoryBarrier2[]
    spans = @NamedTuple{buf::Any, handle::UInt64, off::UInt64, len::UInt64,
                        ss::Any, sa::Any, ds::Any, da::Any}[]
    for t in ts
        src_stage = reduce(|, (stages(be, u, Src()) for u in t.waits))
        src_access = reduce(|, (access(be, u, Src()) for u in t.waits))
        dst_stage = stages(be, t.to, Dst())
        dst_access = access(be, t.to, Dst())
        r = t.resource == 0 ? nothing : get(g.ids.by_id, t.resource, nothing)
        # A resource that can be renamed gets a global barrier instead of one
        # scoped to its buffer. `st.buf[].buffer` below is baked here, at compile
        # time, and `rename!` points the resource at a *different* store — so a
        # scoped barrier would name the store the plan was compiled with and
        # cover none of the memory that is actually read. Widening loses the span
        # scoping for these resources and is the only correct answer: a handle
        # cannot be baked for something that moves.
        st = (r === nothing || renameable(g, r)) ? nothing : storage(r)
        if st === nothing
            push!(mem, VK._MemoryBarrier2(;
                src_stage_mask = src_stage, src_access_mask = src_access,
                dst_stage_mask = dst_stage, dst_access_mask = dst_access))
        else
            off, len = barrierspan(r, st)
            b = barrierbuffer(r, st)
            push!(spans, (buf = b, handle = UInt64(b.vks), off = off, len = len,
                          ss = src_stage, sa = src_access, ds = dst_stage, da = dst_access))
        end
    end

    # Adjacent segments that ask for the same thing become one barrier — the
    # merge half of the interval map, without which a whole-buffer usage of a
    # buffer sliced in four places emits four entries describing one span. Only
    # touching spans with identical masks merge; anything else stays its own
    # barrier, which is the whole point of scoping them.
    sort!(spans, by = s -> (s.handle, s.off))
    bufs = VK._BufferMemoryBarrier2[]
    i = 1
    while i <= length(spans)
        s = spans[i]
        off, len = s.off, s.len
        j = i + 1
        while j <= length(spans)
            t = spans[j]
            (t.handle == s.handle && t.off == off + len &&
             t.ss == s.ss && t.sa == s.sa && t.ds == s.ds && t.da == s.da) || break
            len += t.len
            j += 1
        end
        push!(bufs, VK._BufferMemoryBarrier2(
            VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
            s.buf, off, len;
            src_stage_mask = s.ss, src_access_mask = s.sa,
            dst_stage_mask = s.ds, dst_access_mask = s.da))
        i = j
    end
    VK._DependencyInfo(mem, bufs, [])
end

function emit_pass_barrier!(bq, dep)
    dep === nothing && return false
    batch = bq.active_batch
    batch === nothing && (batch = ensure_active_batch!(bq))
    VK._cmd_pipeline_barrier_2(batch.cmd_buf, dep)
    true
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
function handover!(pl::LavaPlan, bq)
    # `foldl`, not `any`: short-circuiting would skip recording this plan as the
    # runner of the arenas after the first hit, and the next run would then think
    # it was taking over from someone else.
    pool = pool(pl.graph.dev)
    # `foldl`, not `any`: short-circuiting would skip recording this plan as the
    # runner of the arenas after the first hit, and the next run would then think
    # it was taking over from itself.
    handed = foldl(pl.arenas; init = Any[]) do acc, ar
        takeover!(pool, ar, pl) && push!(acc, ar)
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
        reg = arenaof(pool, ar).region
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
    emit_pass_barrier!(bq, dep)
end

# `peakbytes`/`naivebytes` are NOT defined here. `LavaPlan <: Plan` and
# both read a field the same way in every backend, so core owns them.

"""
The state a compilation carries between phases. Every field is written by exactly
one phase and read by later ones, which is what lets a phase be run alone.
"""
mutable struct Compile <: Compilation
    graph::LavaGraph
    alias::Bool
    coalesce::Bool
    policy::Policy
    # What the backend-independent phases compute. Held rather than spread over
    # fields here, so those phases can live in core and read one thing.
    analysis::Analysis
    transitions::Vector{Transition}            # Barriers
    prepass::IdDict{Pass,Vector{Transition}}
    passes::Vector{PassPlan}                          # Pipelines
    pipelines::Set{Any}
end

Compile(g::LavaGraph; alias = true, coalesce = true, policy = Overlap()) =
    Compile(g, alias, coalesce, policy, Analysis(), Transition[],
            IdDict{Pass,Vector{Transition}}(), PassPlan[], Set{Any}())

# What core's phases ask of a compilation context, and it is now only what
# DIFFERS from every other backend. The seven field accessors — `analysis`,
# `passes`, `policy`, `alias`, `transients`, `transientbyid`, `device` — were
# written verbatim here and in `MantleHostExt`, so they come from
# `Compilation` and neither writes them.
#
# These three stay because they are genuinely this backend's: how a pass records
# its usages, whether two ids can name the same bytes (this backend HAS slices,
# so it is not `a == b`), and which pool the memory comes from.
usages(p::Pass) = p.usages
overlapping(c::Compile, a::Int, b::Int) = overlapping(c.graph, a, b)
pool(c::Compile) = c.graph.dev.pool
nbytes(t::TransientResource) = nbytes(t)
alignment(t::TransientResource) = alignment(t)
arena(t::TransientResource) = arena(t)
describe(t::TransientResource) = describe(t)

# ── Dag ───────────────────────────────────────────────────────────────────────
writes_it(U) = writes(U)

"""
Whether two resource ids can name the same bytes.

Equal ids do. So does a slice against its own parent, and two slices of one
parent whose ranges intersect — those are different ids, and a scheduler told
they are unrelated is free to reorder two passes that write the same memory.
"""
function overlapping(g::LavaGraph, a::Int, b::Int)
    a == b && return true
    ra, rb = get(g.ids.by_id, a, nothing), get(g.ids.by_id, b, nothing)
    (ra isa BufferRange || rb isa BufferRange) || return false
    # `get`, not `resourceid`: this answers a question and must not hand out an
    # id doing it. A slice's parent is always registered first — `use` touches it
    # before `slice` is reached — so the fallback is unreachable, but a query that
    # can grow `by_id` while the compiler is indexing by id is not worth leaving
    # to that invariant holding.
    pa = ra isa BufferRange ? get(g.ids.ids, ra.parent, 0) : a
    pb = rb isa BufferRange ? get(g.ids.ids, rb.parent, 0) : b
    (pa == 0 || pb == 0) && return false
    pa == pb || return false
    # A whole-resource usage covers every slice of it.
    (ra isa BufferRange && rb isa BufferRange) || return true
    !isempty(intersect(ra.range, rb.range))
end

"""
An edge from i to j when they share a resource and at least one writes it.

Read-after-read is deliberately not an edge: two passes that only read the same
thing may run in either order, which is the freedom the scheduler spends.
"""
# ── Place ─────────────────────────────────────────────────────────────────────
"""Back an arena. Buffers get a Lava array; images get raw device memory."""
# The arena is one allocation shared by transients of several element types, so
# its usage bits are the union of what they ask for.
"""
The buffer arena's block: one raw `vkAllocateMemory` with a `VkBuffer` bound into
it, and the buffer's device address.

Deliberately NOT a `LavaArray`. That would put Mantle's suballocation on top of
Lava's pool — two allocators with the lower one invisible — and it would drag in
a `DataRef` whose finalizer frees memory Mantle owns. Nothing here finalizes and
nothing refcounts: the Block owns the memory, the Pool owns the Block, `trim!`
frees.
"""
struct BufferBlock
    buffer::Any
    memory::Any
    address::UInt64
    bytes::Int
    # A `DataRef` whose releaser DOES NOTHING. This is Lava's shape of the
    # `unsafe_wrap(..., own = false)` hatch every GPU array package provides for
    # foreign memory: a transient's `storage` is a `LavaArray` view over it, so
    # host-side operations (copies, library calls) work — while the free stays
    # Mantle's, because the Block owns the memory and `trim!` frees.
    ref::Any
end

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
                                   0, nothing, nothing, BUF_STATE_ALIVE, 0, false, dev.ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, UInt64(addr), Int(req.size), ref)
end

rawalloc(dev::LavaDevice, ::Images, bytes::Int, bits) =
    device_memory(dev.ctx, max(bytes, 1), bits)

rawfree(::LavaDevice, mem) = nothing   # Lava frees its own; see the staging note

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

"""Give a placed transient its storage."""
function materialize!(t::TransientBuffer, blk::BufferBlock, offset::Int)
    t.block, t.offset = blk, offset
    return t
end

function materialize!(t::TransientImage{T}, slab, offset) where {T}
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

"""
What a pass does to one resource, or `nothing` if it does not name it.

A slice counts as naming its parent. The handover asks this about a *transient*,
and a pass that names only a slice of one would otherwise answer `nothing` — on
which the caller skips the barrier entirely, which is the hazard no per-resource
sequence can see going missing without a word.
"""
function usage_of(p::Pass, id::Int, g::LavaGraph)
    for (rid, U) in p.usages
        overlapping(g, rid, id) && return U
    end
    nothing
end

"""
Every id one resource is tracked under: itself, plus each slice of it.

Built once per compile rather than rediscovered per handover. `lastuses` used to
find these by scanning every tracked state and asking `overlapping`, which is
O(resources) inside a per-pass loop — invisible on a twenty-pass render graph and
1.75 s of a 2 s compile at fourteen hundred, which is the scale a model graph
arrives at. Resources without slices get an empty entry and the O(1) path.
"""
function sliceindex(g::LavaGraph)
    idx = Dict{Int,Vector{Int}}()
    for ((pid, _), v) in g.views
        push!(get!(() -> Int[], idx, pid), resourceid(g, v))
    end
    idx
end

"""
Everything the old tenant was last doing, across however many ids it is tracked
under. Whole and sliced usages of one transient live under different ids, and the
handover has to wait for all of them, not for whichever the transient itself
happens to be keyed by.
"""
function lastuses(states::Dict{Int,ResourceState}, slices::Dict{Int,Vector{Int}},
                  id::Int)
    out = Type[]
    take(k) = begin
        st = get(states, k, nothing)
        st === nothing || st.current === nothing || st.current in out ||
            push!(out, st.current)
    end
    take(id)
    for sid in get(slices, id, ())
        take(sid)
    end
    out
end

# ── Barriers ──────────────────────────────────────────────────────────────────
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
function run!(::Pipelines, c::Compile)
    g = c.graph
    argcursor = 0        # every draw's argument block, laid out once
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
        cps = CompiledDispatch[]
        # A `:custom` pass carries a closure here, not a `Dispatch`. There is
        # nothing to compile and no arguments to lay out: whatever it launches,
        # it launches through the backend at record time.
        for d in (p.kind === :custom ? () : p.dispatches)
            cd = compile_dispatch(g.dev, d, argcursor)
            # Not into `c.pipelines`: that set answers how many shaders the draws
            # resolved to — two draws sharing one is the thing worth counting —
            # and a dispatch's pipeline is held by its `CompiledDispatch` anyway.
            push!(cps, cd)
            argcursor += argalign(cd.argsize)
        end
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
function compile_dispatch(dev::LavaDevice, d::Dispatch, argoff::Int)
    obj = kernelfor(d.kernel, d.group)
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
                     tlas !== nothing, argoff, launch.total_size)
end

# An ndrange fixed when the graph was built, or one that is read per frame — the
# same distinction `drawover` makes for a draw's vertex count.
dispatchrange(n::Integer) = n
dispatchrange(t::Tuple) = t
dispatchrange(x) = count(x)

# A `DeviceRange` is never read here: the count lives on the device and reading
# it would be the host readback the whole mechanism exists to avoid. The kernel
# is compiled against a CEILING so `__validindex` lets every thread through, and
# the real bound is the count the GPU reads at dispatch time — the kernel's own
# `i <= n` check does the rest. Same contract as `ka_launch_indirect!`,
# which is what records it.
const INDIRECT_CEILING = 1024 * 1024
dispatchrange(r::DeviceRange) = something(r.max, INDIRECT_CEILING)

Plan(g::LavaGraph; coalesce::Bool = true, alias::Bool = true,
            profile::Bool = false, policy::Policy = Overlap()) =
    let c = compile!(Compile(g; alias, coalesce, policy))
        let a = analysis(c)
            pl = LavaPlan(g, c.transitions, c.passes, c.pipelines, a.regions, a.arenas,
                          a.offsets, a.peak, a.naive,
                          profile ? Profiler(g.dev.ctx, c.passes) : nothing,
                          alias, coalesce, policy, ArgMemory(g.dev, c.passes), nothing)
            # After construction, because a plan cannot be a tenant before it is a
            # plan — and the arena it was just placed into may grow for the NEXT
            # plan, which is when this registration earns its keep.
            for ar in a.arenas
                tenant!(pool(g.dev), ar, pl)
            end
            pl
        end
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
function refit!(pl::LavaPlan)
    # Not `any`, which short-circuits: the first transient that moved would be
    # the only one refitted.
    moved = false
    for t in pl.graph.transients
        moved |= refit!(t)
    end
    moved || return false
    c = compile!(Compile(pl.graph; pl.alias, pl.coalesce, pl.policy))
    pl.transitions, pl.passes, pl.pipelines = c.transitions, c.passes, c.pipelines
    let a = analysis(c)
        pl.slabs, pl.arenas, pl.offsets = a.regions, a.arenas, a.offsets
        pl.peak, pl.naive = a.peak, a.naive
        # Re-registered: the recompile may have been placed into a different set
        # of arenas, and `tenant!` is idempotent for the ones it was already in.
        for ar in a.arenas
            tenant!(pool(pl.graph.dev), ar, pl)
        end
    end
    # A recompile can change the draws and therefore the layout, so the argument
    # memory is laid out again with them. The old slots are still being read by
    # frames in flight; `sync_swapchain!` waited for the device before any of
    # this, so letting them go here is safe.
    pl.args = ArgMemory(pl.graph.dev, c.passes)
    return true
end

npipelines(pl::LavaPlan) = length(pl.pipelines)

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

"""
Nanoseconds between a pass's two timestamps, or `nothing` if the pair cannot be
from one frame.

The queries are `UInt64`, so an end that precedes its start does not come out
negative — it wraps to about 1.8e19 ns, which `timings` then reports as a
hundred billion milliseconds. That happens: the pool is reset at the head of
every frame's recording, and a read that races the reset can take the start word
from one frame and the end word from the next, with both availability bits set.
There is no time to report for such a pair, so it is dropped rather than
averaged in.
"""
elapsed(lo::UInt64, hi::UInt64, period) = hi < lo ? nothing : Float64(hi - lo) * period

"""
    timings(plan)

Per pass, median host recording time and median GPU time over the samples kept.
"""
function timings(pl::LavaPlan)
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
Every attachment has to cover the render area, and a window changes size under a
plan compiled for the old one: the swapchain follows a resize, a transient sized
when the graph was built does not. Vulkan calls the result undefined
(VUID-VkRenderingInfo-pNext-06079) and RADV draws it anyway, so without this it is
a wrong picture rather than a message.
"""
function checkextents(pl::LavaPlan)
    for pp in pl.passes
        p = pp.pass
        p.kind === :render || continue
        want = target_extent(first_target(p))
        for t in (p.targets..., p.depth)
            t === nothing && continue
            e = target_extent(t)
            (e.width < want.width || e.height < want.height) &&
                error("pass \"$(p.name)\": the render area is $(want.width)x$(want.height) " *
                      "but an attachment is $(e.width)x$(e.height). A resize follows the " *
                      "swapchain and not a transient sized when the graph was built — " *
                      "rebuild the graph and the plan at the new size.")
        end
    end
    nothing
end

"""
    run!(plan; barriers = :derived)

`:derived` emits the ordering the declarations call for and suppresses Lava's
automatic per-dispatch barrier. `:backend` does the opposite and exists so the
two can be measured against each other rather than argued about.

`:both` emits the derived barriers *and* leaves the automatic one in place. It is
a diagnostic, and the only one that separates the two ways a `custom!` graph can
be wrong: a result that is correct under `:both` and wrong under `:derived` says
the declared set is incomplete, while one that is wrong under both says the
declarations are right and something is reading the wrong bytes.
"""
baked(pl::LavaPlan) = pl.baked !== nothing

"""A baked plan's recording holds the addresses its region has today, so the
arena it is placed in can no longer grow. See `remappable`."""
remappable(pl::LavaPlan) = pl.baked === nothing

function bake!(pl::LavaPlan)
    pl.baked === nothing || return pl      # idempotent; re-baking would strand the old one
    g = pl.graph
    isempty(g.surfaces) || throw(ArgumentError(
        "bake!: this plan draws to a surface. A swapchain image is a different image " *
        "every frame and a recording names one, so a windowed plan needs a recording " *
        "per swapchain image — which is not built yet. Headless plans bake today."))
    pl.profiler === nothing || throw(ArgumentError(
        "bake!: profiling and baking do not combine yet. `timings` measures host " *
        "recording per pass, and a baked plan does not record — the numbers would be " *
        "the ones from the capture, reported forever. Build the plan without " *
        "`profile = true`, or do not bake it."))
    bq = g.dev.bq
    refit!(pl)
    checkextents(pl)
    # The slot this recording names for the rest of its life. Taken once, here,
    # because `slotbase(am)` is folded into every address the command buffer
    # holds: rotating afterwards would aim the replay at a slot something else
    # is free to write.
    nextslot!(pl.args, bq)
    pl.baked = capture(bq) do
        concurrent_dispatch_group() do
            record!(pl, bq; derived = true, suppress = true, updates = false)
        end
    end
    pl
end

function run!(pl::LavaPlan; barriers::Symbol = :derived)
    checklive(pl, pl.slabs, length(pl.graph.transients))
    # Anything dropped without a `free!` goes back here, one submission boundary
    # after it was dropped. Cheap and a no-op when nothing was — see
    # `reclaim!`. Here rather than in a user's frame loop because a
    # renderer that has to remember to call it is one that stops reclaiming the
    # day someone forgets, which is the failure this exists to remove.
    reclaim!(pool(pl.graph.dev), pl.graph.dev)
    barriers in (:derived, :backend, :both) ||
        throw(ArgumentError("barriers must be :derived, :backend or :both, got $barriers"))
    g = pl.graph
    bq = g.dev.bq
    # Before this frame overwrites them, and without waiting: see `collect!`.
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
    if pl.baked !== nothing
        # A refit re-places every transient, so the recording names storage that
        # no longer exists. Nothing tracking can move in a headless plan today,
        # which is why this is an assertion rather than a re-bake.
        moved && throw(ArgumentError(
            "run!: a transient moved under a baked plan, so its recording names " *
            "storage that has been replaced. Re-`Plan` and `bake!` again."))
        # Updates first and fresh — `replay!` closes any batch still recording,
        # so the copies land ahead of the replay in queue order, which is the
        # order the derived barriers inside the recording were built for.
        record_updates!(pl, bq)
        # A replay writes this arena's bytes like any other run, so it has to
        # claim them — even though it cannot emit a barrier of its own, since a
        # recording is frozen. Skipping the claim leaves the arena naming
        # whoever RECORDED last, and the next tenant then sees itself there and
        # emits nothing: a handover away from a baked plan with no barrier at
        # all. (The baked plan taking over from someone else is still
        # unbarriered — that is what `bake!`ing into a shared arena costs, and
        # `remappable` already refuses the growth case.)
        let pool = pool(pl.graph.dev)
            for ar in pl.arenas
                takeover!(pool, ar, pl)
            end
        end
        replay!(pl.baked)
        return nothing
    end
    for s in g.surfaces
        acquire_next_image!(s.win)
    end
    # One slot of the plan's argument memory per frame, reused only once the GPU
    # has passed the frame that last used it.
    nextslot!(pl.args, bq)
    if barriers === :derived
        # The group is a scope rather than a flag, so nothing leaks past here.
        concurrent_dispatch_group() do
            record!(pl, bq; derived = true, suppress = true)
        end
    else
        record!(pl, bq; derived = barriers === :both, suppress = false)
    end
    # What this frame signals covers everything written into the slot. A split
    # mid-frame makes later batches with higher values, and the last one covers
    # them all, so reading it after recording is right.
    pl.args.signal[pl.args.slot] = ensure_active_batch!(bq).signal_value
    for s in g.surfaces
        present_frame!(bq, s.win)
    end
    nothing
end

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
function profiled!(f, pl::LavaPlan, bq, i::Integer)
    prof = pl.profiler
    prof === nothing && return f()
    cmd = ensure_active_batch!(bq).cmd_buf
    VK.cmd_write_timestamp(cmd, VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                    prof.pool, UInt32(2i - 2))
    t0 = time_ns()
    r = f()
    sample!(prof.host_ns[i], Float64(time_ns() - t0))
    cmd = ensure_active_batch!(bq).cmd_buf   # a pass may have split the batch
    VK.cmd_write_timestamp(cmd, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                    prof.pool, UInt32(2i - 1))
    prof.pending = true
    r
end

function record!(pl::LavaPlan, bq; derived::Bool = true, suppress::Bool = derived,
                 updates::Bool = true)
    g = pl.graph
    # Before anything this plan records: the pool it is about to write may still
    # be being read by whichever plan ran last.
    handover!(pl, bq)
    if pl.profiler !== nothing
        VK.cmd_reset_query_pool(ensure_active_batch!(bq).cmd_buf,
                                         pl.profiler.pool, UInt32(0),
                                         UInt32(pl.profiler.nslots))
    end
    for (i, pp) in enumerate(pl.passes)
        # `bake!` records everything except this, and `run!` records this alone
        # before each replay. An update writes through a fresh store on the
        # rename route, so the copy's destination handle is not the one the
        # recording would hold — see `renameable`.
        (!updates && pp.pass.kind === :update) && continue
        profiled!(pl, bq, i) do
            record_pass!(g, bq, pp, derived, pl.args; suppress)
        end
    end
    nothing
end

"""The update passes alone, recorded fresh in front of a replay."""
function record_updates!(pl::LavaPlan, bq)
    for pp in pl.passes
        pp.pass.kind === :update || continue
        record_pass!(pl.graph, bq, pp, true, pl.args; suppress = true)
    end
    nothing
end

"""One pass: its barriers, then whatever its kind does."""
function record_pass!(g::LavaGraph, bq, pp::PassPlan, derived::Bool, am::ArgMemory;
                      suppress::Bool = derived)
    p, cds = pp.pass, pp.draws
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

    # Image barriers are emitted in both modes. `:backend` hands the buffer
    # hazards to Lava's automatic per-dispatch barrier, but nothing in Lava
    # knows what layout a graph's render target should be in, so dropping
    # these would not be a comparison, it would be a broken frame.
    for b in pp.images
        emit_barrier!(bq, b)
    end
    derived && emit_pass_barrier!(bq, pp.barrier)

    if p.kind === :compute
        base = slotbase(am)
        # A dispatch sized on the device costs a prepare kernel and a barrier the
        # command processor's read of the count depends on, and that barrier is
        # forced — it is the one hazard a graph cannot derive away, since the
        # prepare is the backend's own machinery rather than a pass. Left per
        # dispatch it serialises a pass's dispatches whatever the schedule said
        # they could do: twelve per-material shading kernels become twelve
        # barriers.
        #
        # So the pass's prepares are fused into one and the dispatches share the
        # single barrier behind it. The prepares are still ordered after this
        # pass's derived barrier, which is what puts them after whoever wrote the
        # counts; within a pass the dispatches are independent by construction,
        # which is the same assumption that already lets them run without
        # barriers between them.
        if pp.indirect
            concurrent_indirect_group(bq) do
                for d in pp.dispatches
                    record_dispatch!(bq, d, am, base)
                end
            end
        else
            for d in pp.dispatches
                record_dispatch!(bq, d, am, base)
            end
        end
        return nothing
    elseif p.kind === :custom
        # The barriers this pass declared are already emitted. Open the batch so
        # the body records into this frame's command buffer rather than opening
        # one of its own, then get out of the way.
        ensure_active_batch!(bq)
        # Two launches inside one body may depend on each other — a two-pass
        # reduction, a split-K matmul, an operator with a scratch buffer — and
        # nothing declared to the graph says so, because the declaration is about
        # what the pass touches and not about how. So the surrounding concurrent
        # group is lifted and Lava's automatic barrier orders the body's own
        # launches, while the *first* of them skips it: that one is the hazard
        # this pass just emitted a derived barrier for.
        exclusive_dispatch_group() do
            bq.next_skip_barrier = suppress
            for body in p.dispatches
                body()
            end
            # A body that recorded no dispatch at all leaves the one-shot armed,
            # and the next pass's first launch would consume it without anyone
            # having decided that.
            bq.next_skip_barrier = false
        end
        return nothing
    elseif p.kind === :update
        # Before the writes, not after: a rename takes a store from the free
        # list, so anything the GPU finished with has to be back on it first.
        # Recycling afterwards leaves last frame's store invisible to this
        # frame's take, and the resource ping-pongs across three buffers
        # instead of two.
        recycle!(g.recycler, bq)
        applyupdates!(g.updates) do r, data
            write_update!(g, bq, r, data)
        end
        return nothing
    elseif p.kind === :copy
        src, dst = first_target(p), p.dst
        copy_image_to_buffer!(bq, storage(dst), target_image(src),
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
    vk_begin_pass!(bq,
                        VK.ImageView[target_view(t) for t in p.targets],
                        VK.Image[target_image(t) for t in p.targets],
                        target_extent(first_target(p));
                        clear_color = map(clearvalue, p.loads),
                        load_op = map(loadop, p.loads),
                        depth_view = p.depth === nothing ? nothing : target_view(p.depth),
                        depth_clear = p.depth === nothing ? nothing : depthclear(p.depth_load),
                        depth_load_op = p.depth === nothing ? nothing : loadop(p.depth_load),
                        transition = false)
    # Viewport and scissor are dynamic pipeline state. vk_draw! sets them for
    # you; vk_draw_in_pass! only does so when passed, and omitting them
    # rasterizes nothing without raising anything.
    ext = target_extent(first_target(p))
    vk_set_viewport!(bq,
        VK.Viewport(0f0, 0f0, Float32(ext.width), Float32(ext.height), 0f0, 1f0),
        VK.Rect2D(VK.Offset2D(0, 0), ext))
    base = slotbase(am)
    for d in cds
        record_draw!(bq, d, am, base)
    end
    vk_end_pass!(bq)
    nothing
end

"""
One draw: write its arguments into the plan's slot and record it.

Its own function because a pass's draws are differently parameterised, so the
loop dispatches once per draw and everything inside is concrete — including
`devargs(ad, rawargs(d.args))`, which allocated a boxed tuple per draw per frame while it
was inlined into the loop.

Nothing is allocated and nothing is pinned: the memory belongs to the plan, and
every resource the arguments name is reachable from the plan for as long as it
lives.
"""
function record_draw!(bq, d::CompiledDraw, am::ArgMemory, base::Int)
    off = base + d.argoff
    info = d.shader.push_info
    pack_args_direct!(bq, am.ptr + off, am.address + off, info.arg_offsets,
                           info.arg_buffer_size, info.byval_llvm_sizes,
                           devargs(adaptor(bq), rawargs(d.args)))
    # No viewport, no scissor, no pin: the pass set the first two once, and the
    # plan holds the pipeline for longer than any frame.
    emit_draw!(bq, d.compiled, d.count, am.address + off)
end

# Where the counts come from, decided once by type rather than per frame by a
# branch. The indirect one records the same command whatever the numbers are,
# which is why nothing has to be read back to record a frame.
emit_draw!(bq, pipe, n::Integer, addr::UInt64) =
    vk_draw_in_pass!(bq, pipe, n; push_bda = addr, pin = false)
emit_draw!(bq, pipe, x, addr::UInt64) =
    vk_draw_in_pass!(bq, pipe, count(x); push_bda = addr, pin = false)
emit_draw!(bq, pipe, c::Commands, addr::UInt64) =
    vk_draw_indirect_in_pass!(bq, pipe, storage(c.resource);
                                   push_bda = addr, pin = false)

"""
One dispatch: the same two steps as a draw, into the same memory.

Nothing here can compile, allocate, or take a buffer from Lava's argument pool.
An ndrange that has not moved reuses the iteration plan the pipeline was built
with; one that has costs a lookup keyed on the shape, which is not keyed on the
world and so still cannot reach the compiler.
"""
function record_dispatch!(bq, d::CompiledDispatch{K,A,I}, am::ArgMemory, base::Int) where {K,A,I}
    nd = dispatchrange(d.ndrange)
    it = nd == d.nd0 ? d.iter :
         get_or_build_iter_plan(d.obj, nd, nothing, bq.ctx::VkContext)::I
    it.nblocks == 0 && return nothing
    tlas = packdispatch!(bq, d, am, base, it)
    recordlaunch!(bq, d.ndrange, lp_of(d), am.address + base + d.argoff, it, tlas)
    nothing
end

lp_of(d::CompiledDispatch) = d.launch

"""
Write one dispatch's arguments into the plan's slot, and answer with the
acceleration structure they name.

Separate from recording because a **baked** plan has to do exactly this and
nothing else: its command buffer holds the ADDRESS of the slot, so a value that
moved is a write to host-mapped memory rather than a new recording. See
[`rebind!`](@ref).
"""
function packdispatch!(bq, d::CompiledDispatch, am::ArgMemory, base::Int, it)
    off = base + d.argoff
    lp = d.launch
    raw = rawargs(d.args)
    args = devargs(adaptor(bq), raw)
    pack_args_direct!(bq, am.ptr + off, am.address + off, lp.offsets,
                           lp.arg_buffer_size, lp.byval_sizes,
                           (d.kernel, it.ka_ctx, args...))
    return d.tlas ? find_tlas_in_args(raw) : nothing
end

"""A plan with a `custom!` pass cannot be rebound: the body packs its own
arguments as it runs, and a baked plan never runs it again."""
rebindable(pl::LavaPlan) = !any(pp -> pp.pass.kind === :custom, pl.passes)

"""
    rebind!(plan) -> plan

Re-read the arguments a **baked** plan's work was given and write the current
values into the recording's argument memory.

A baked plan does not record, so the values packed at `bake!` are the ones it
replays — for ever, and silently. A frozen sample index renders the same sample
every time and converges to a picture that looks plausible and is wrong, which is
the failure this exists to prevent. Anything given as a `Ref` moves; this is what
makes it move again once the plan is baked.

It writes bytes and records nothing: the command buffer holds the address of the
slot, and the slot is host-mapped, so this is a pack per dispatch and no command
buffer is touched. On an unbaked plan it is a no-op, because every `run!` packs
already.

**The ordering is the caller's.** A baked plan names one argument slot for its
whole life, so rewriting it while an earlier replay is still reading it is a
race — this cannot wait on your behalf without turning every rebind into a device
drain. Call it where the device is known to be past that replay; a renderer that
already synchronises once per sample has such a point.
"""
function rebind!(pl::LavaPlan)
    pl.baked === nothing && return pl
    rebindable(pl) || throw(ArgumentError(
        "rebind!: this plan has a `custom!` pass, whose body packs its own " *
        "arguments while it runs — and a baked plan never runs it again, so the " *
        "values it replays are the ones captured at `bake!`. Returning quietly " *
        "would leave them stale and produce a plausible wrong result. Ask " *
        "`rebindable(plan)` before baking anything whose arguments move."))
    bq = pl.graph.dev.bq
    am = pl.args
    base = slotbase(am)
    for pp in pl.passes
        for d in pp.dispatches
            nd = dispatchrange(d.ndrange)
            it = nd == d.nd0 ? d.iter :
                 get_or_build_iter_plan(d.obj, nd, nothing, bq.ctx::VkContext)
            it.nblocks == 0 && continue
            packdispatch!(bq, d, am, base, it)
        end
        for dr in pp.draws
            off = base + dr.argoff
            info = dr.shader.push_info
            pack_args_direct!(bq, am.ptr + off, am.address + off, info.arg_offsets,
                                   info.arg_buffer_size, info.byval_llvm_sizes,
                                   devargs(adaptor(bq), rawargs(dr.args)))
        end
    end
    return pl
end

"""
Record the launch itself. A host-side ndrange dispatches directly; a
[`DeviceRange`](@ref) converts the element count to workgroup counts on the
device and dispatches indirectly off that.

The conversion is a kernel, so it cannot live in Mantle core — but the CONTRACT
does: core hands over an element count and knows nothing about workgroups, and
the graph orders whatever wrote the count before this dispatch because
`indirectcount!` registered it as an `Indirect` read. That barrier is the one
Hikari places by hand today, and getting it wrong is what
`concurrent_indirect_group` had to be taught about the hard way.
"""
recordlaunch!(bq, ::Any, lp, argaddr, it, tlas) =
    vk_dispatch!(bq, lp.pipeline, argaddr, it.block_dims, tlas)

function recordlaunch!(bq, r::DeviceRange, lp, argaddr, it, tlas)
    indirect = get_indirect_buffer(bq)
    deferred = bq.deferred_indirect
    if deferred === nothing
        fast_prepare_indirect!(bq, indirect, storage(r.count), prod(it.ws_3d))
        vk_dispatch_indirect!(bq, lp.pipeline, argaddr, indirect, tlas)
    else
        # Inside the pass's group: hand over the slot and the count, and let the
        # flush emit one prepare for all of them. Same protocol Lava's own
        # `ka_launch_indirect!` uses, so there is one deferral format rather than
        # two that have to agree.
        push!(deferred, (bq, lp.pipeline, argaddr, indirect, tlas,
                         storage(r.count), prod(it.ws_3d)))
    end
    nothing
end

"""
Give this plan's regions back to the pool. The argument memory and the
pipelines are ordinary Lava objects — the GC reclaims those; the regions are
the thing only an explicit call can return, because nothing here finalizes.

No precondition: the regions are retired, so a plan freed immediately after its
last `run!` — the ordinary case, with its recording still in flight — is fine.
"""
function free!(pl::LavaPlan)
    # The capture too, and before the regions: it holds the argument memory its
    # recording points at, plus a reference to every resource the recording
    # names. Dropping the plan alone would leave both to the GC, which does not
    # know it is holding device memory.
    if pl.baked !== nothing
        release!(pl.baked)
        pl.baked = nothing
    end
    giveup!(pool(pl.graph.dev), pl.slabs, pl.arenas, pl)
end

"""Re-materialise this plan's transients of `kind` into the arena's new region.
Its own offsets are unaffected by the arena moving; only the base changed."""
function remap!(pl::LavaPlan, kind, region)
    ts = pl.graph.transients
    for (i, t) in enumerate(ts)
        arena(t) == kind || continue
        materialize!(t, memoryof(region), offset(region) + pl.offsets[i])
    end
    return pl
end

Base.close(::LavaPlan) = nothing
Base.isopen(s::LavaSurface) = isopen(s.win)
