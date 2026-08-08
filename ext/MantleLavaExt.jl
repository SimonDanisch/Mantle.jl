"""
The Lava backend: Mantle owns lifetimes, the graph and the plan; Lava supplies
the Vulkan mechanism it drives.
"""
module MantleLavaExt

using Mantle
using Mantle: Usage, Storage, ColorAttachment, Depth, Sampled, Present, Undefined, Vertices,
              CopySrc, CopyDst, ReadOnly, WriteOnly, ReadWrite, NoAccess, ImageKind, BufferKind,
              Src, Dst, transitions
import Lava
import Mantle: storage, Buffer, Scalar, Surface, Attribute, draw!, dispatch!, render!, compute!,
               run!, npipelines, stride, count

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
mutable struct ArenaPool
    memory::Any                  # the backing allocation, or nothing before the first reserve
    bytes::Int                   # what it is sized to
    extra::UInt32                # union, for a buffer arena
    typebits::UInt32             # intersection, for an image arena
    tenants::Vector{WeakRef}     # the LavaPlans holding space here, for re-materialising
end

ArenaPool() = ArenaPool(nothing, 0, UInt32(0), typemax(UInt32), WeakRef[])

struct LavaDevice <: Mantle.Device
    ctx::Any
    bq::Any
    # Keyed by the arena marker (`Buffers()` / `Images()`), which is a singleton,
    # so this is a two-entry dictionary consulted once per compile — never on the
    # frame path, which is why `Any` here costs nothing.
    pools::Dict{Any,ArenaPool}
end

Mantle.Device(::typeof(Lava)) =
    LavaDevice(Lava.vk_context(), Lava.vk_context().default_bq, Dict{Any,ArenaPool}())
Mantle.Device() = Mantle.Device(Lava)

pool!(dev::LavaDevice, a) = get!(ArenaPool, dev.pools, a)

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
function Mantle.capacity(dev::LavaDevice)
    heaps = Lava.probe_device_memory_budget(dev.ctx)
    total = 0
    for h in heaps
        h.device_local || continue
        total += h.budget > 0 ? h.budget : h.size
    end
    total
end

"""The KernelAbstractions backend, for kernels the graph does not own."""
Mantle.backend(::LavaDevice) = Lava.LavaBackend()

"""
What this device can do, as Mantle's portable record rather than Lava's.

A field-by-field copy of two structs that happen to agree today, and deliberately
not an alias. Lava's `DeviceCaps` is Lava's to change; Mantle's is the one kernels
are written against, and the day a backend reports something the other does not
have, this conversion is where that shows up rather than in every kernel.
"""
Mantle.caps(dev::LavaDevice) = mantlecaps(Lava.caps(dev.ctx))
Mantle.caps(b::Lava.LavaBackend) = mantlecaps(Lava.caps(b))

mantlecaps(c) = Mantle.DeviceCaps(c.coopmat, c.tile, c.subgroup, c.coopmatsubgroup,
                                  c.sharedbudget, c.workgrouplimit, c.cores, c.warps)

# ── window ────────────────────────────────────────────────────────────────────
"""
A window, which is Mantle's because the frame loop is: `isopen` has to be a
predicate rather than a thing that also pumps events, and `run!` is the one call
per frame that can pump them.
"""
struct LavaWindow <: Mantle.Window
    win::Lava.RenderWindow
end

Mantle.Window(width::Integer, height::Integer; title::AbstractString = "", vsync::Bool = false) =
    LavaWindow(Lava.RenderWindow(width, height; title = String(title), vsync))

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
Mantle.screenshot(w::LavaWindow) = Lava.readback_window(w.win)

# ── persistent resources ──────────────────────────────────────────────────────
"""
Length and capacity are separate so a count that changes every frame never
reallocates. `store` is always at capacity; `len` is what a draw covers.
"""
mutable struct LavaBuffer{T} <: Mantle.Resource
    # Concrete, not `Any`: a rename swaps the store for a fresh one but never for
    # a different type, and `Any` here made every `storage(buf)` in the per-frame
    # argument path return a boxed value.
    store::Lava.LavaArray{T,1}
    len::Int
    capacity::Int
    dev::LavaDevice
end

"""
Usage bits an element type asks for beyond the ordinary ones, which is one type
and one bit: a buffer of draw commands is read by the command processor, and a
buffer without `INDIRECT_BUFFER_BIT` is a validation error at the draw rather
than where it was allocated.
"""
extrausage(::Type) = UInt32(0)
extrausage(::Type{Lava.DrawIndirectCommand}) = UInt32(Lava.Vulkan.BUFFER_USAGE_INDIRECT_BUFFER_BIT)

function Buffer(dev::LavaDevice, data::AbstractVector{T}; capacity = length(data)) where {T}
    cap = max(capacity, length(data))
    store = Lava.LavaArray{T,1}(undef, (cap,); extra_usage = extrausage(T))
    copyto!(store, 1, data, 1, length(data))
    LavaBuffer{T}(store, length(data), cap, dev)
end

Buffer(dev::LavaDevice, ::Type{T}, n::Integer) where {T} =
    LavaBuffer{T}(Lava.LavaArray{T,1}(undef, (Int(n),); extra_usage = extrausage(T)),
                  Int(n), Int(n), dev)

"""One value shared by every element. A distinct type, not a length-1 buffer:
"shared by all" is a claim about meaning, not a length that happens to be one."""
mutable struct LavaScalar{T} <: Mantle.Resource
    store::Lava.LavaArray{T,1}
    dev::LavaDevice
end

Scalar(dev::LavaDevice, x::T) where {T} = LavaScalar{T}(Lava.LavaArray(T[x]), dev)

Base.length(b::LavaBuffer) = b.len
Base.length(::LavaScalar) = 1
Mantle.capacity(b::LavaBuffer) = b.capacity
count(b::LavaBuffer) = b.len
count(::LavaScalar) = 1

stride(::LavaBuffer) = 1
stride(::LavaScalar) = 0

Base.eltype(::LavaBuffer{T}) where {T} = T
Base.eltype(::LavaScalar{T}) where {T} = T

Mantle.update!(b::LavaBuffer, data::AbstractVector) = (Mantle.update!(b, 1:length(data), data); b)
function Mantle.update!(b::LavaBuffer, r::AbstractUnitRange, data::AbstractVector)
    length(r) == length(data) || throw(DimensionMismatch("range $r vs $(length(data)) elements"))
    last(r) <= b.capacity || throw(BoundsError(b, r))
    copyto!(b.store, first(r), data, 1, length(data))
    b.len = max(b.len, last(r))
    b
end
Mantle.update!(b::LavaBuffer, i::Integer, x) = Mantle.update!(b, i:i, [x])
Mantle.update!(s::LavaScalar, x) = (copyto!(s.store, 1, [x], 1, 1); s)

function Base.resize!(b::LavaBuffer, n::Integer)
    n <= b.capacity || throw(ArgumentError("resize beyond capacity is not implemented yet"))
    b.len = Int(n)
    b
end

Base.Array(b::LavaBuffer) = Array(b.store)[1:b.len]

# ── the window ────────────────────────────────────────────────────────────────
"""
The swapchain image is externally indexed: acquire returns whatever the
presentation engine chooses, not `frame % n`. So it is its own type, and `run!`
brackets the acquire, the wait and the present rather than any user code.
"""
struct LavaSurface <: Mantle.Resource
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

target_view(fb::Lava.LavaFramebuffer) = fb.color_view
target_image(fb::Lava.LavaFramebuffer) = fb.color_image
target_extent(fb::Lava.LavaFramebuffer) = Lava.Vulkan.Extent2D(fb.width, fb.height)
target_format(fb::Lava.LavaFramebuffer) = fb.color_format

# What state the target is in when the pass starts. The window arrives from
# acquire; an offscreen target was last read by the copy that took it to the
# window. Both are discarded by a clearing pass, so this only matters when the
# pass loads. A placed target has no contents at all until something writes it,
# and after aliasing it is back in that state, so it starts from Undefined.
initial_usage(::LavaSurface) = Present
initial_usage(::Lava.LavaFramebuffer) = CopySrc

# What state a resource is in when a replay begins. `nothing` means the first use
# establishes it and no transition into it is needed, which is the answer for
# every buffer: buffers have no layout, so there is nothing to transition from.
initial_state(::Any) = nothing
initial_state(s::LavaSurface) = initial_usage(s)
initial_state(fb::Lava.LavaFramebuffer) = initial_usage(fb)
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
    loads::Vector{Mantle.LoadOp}        # render only, one per colour attachment
    depth::Any                          # render only, and only if one was given
    depth_load::Union{Nothing,Mantle.LoadOp}
    dst::Any                            # copy only
    draws::Vector{DrawCall}
    usages::Vector{Pair{Int,Type}}
    dispatches::Vector{Any}
end

Pass(name, kind) = Pass(String(name), kind, Any[], Mantle.LoadOp[], nothing, nothing, nothing,
                        DrawCall[], Pair{Int,Type}[], Any[])

"""The colour attachment a pass configures itself from: extent and viewport are
the same for all of them, so the first answers for the set."""
# The attachment a pass takes its render area from. A depth-only pass — which is
# what a shadow map is — has no colour target, and then the depth one is it.
first_target(p::Pass) = isempty(p.targets) ? p.depth : first(p.targets)

# The load op, lowered. `Discard` is the one that needs saying: it is the only
# way to reach DONT_CARE, and a pass that covers every pixel should not pay to
# load what it is about to overwrite.
loadop(::Mantle.KeepOp) = Lava.Vulkan.ATTACHMENT_LOAD_OP_LOAD
loadop(::Mantle.DiscardOp) = Lava.Vulkan.ATTACHMENT_LOAD_OP_DONT_CARE
loadop(::Mantle.Clear) = Lava.Vulkan.ATTACHMENT_LOAD_OP_CLEAR

clearvalue(::Mantle.LoadOp) = nothing
clearvalue(c::Mantle.Clear) = NTuple{4,Float32}(c.value)

"""What a depth attachment clears to. One number, not four, and `nothing` loads."""
depthclear(::Mantle.LoadOp) = nothing
depthclear(c::Mantle.Clear) = Float32(c.value)

"""
A transient. The compiler owns its interval and its offset; the handle carries no
storage until `Plan` places it. `first`/`last` are pass indices, filled in as the
graph is built, so liveness is not a separate declaration that could disagree.
"""
abstract type Transient <: Mantle.Resource end

mutable struct TransientBuffer{T} <: Transient
    n::Int
    first::Int
    last::Int
    # Nothing until the placer gives it storage; a two-member union splits, an
    # `Any` does not.
    view::Union{Nothing,Lava.LavaArray{T,1}}
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

The `Vulkan.Image` exists from the moment the handle does, because its memory
requirements are what the placer needs and only a real image can be asked. It
costs no memory until bound, and the graph is built once, so this is not a
per-frame cost.
"""
mutable struct TransientImage{T} <: Transient
    width::Int
    height::Int
    format::Lava.Vulkan.Format
    usage::Lava.Vulkan.ImageUsageFlag
    image::Lava.Vulkan.Image
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
target_extent(t::TransientImage) = Lava.Vulkan.Extent2D(t.width, t.height)
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
    Lava.host_buffer(bq, n)
end

"""Take a buffer of exactly `n` bytes, recycled if one is available."""
function take!(r::Recycler, dev::LavaDevice, ::Type{T}, n::Integer) where {T}
    bytes = Int(n) * sizeof(T)
    pool = get(r.free, bytes, nothing)
    if pool !== nothing && !isempty(pool)
        return pop!(pool)::Lava.LavaArray{T,1}
    end
    Lava.LavaArray{T,1}(undef, (Int(n),))
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
    now = Lava.query_timeline(bq)
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

mutable struct LavaGraph <: Mantle.Graph
    dev::LavaDevice
    passes::Vector{Pass}
    surfaces::Vector{LavaSurface}
    transients::Vector{Transient}
    transient_by_id::Dict{Int,Transient}
    by_id::Dict{Int,Any}
    ids::IdDict{Any,Int}
    updates::Vector{Any}
    recycler::Recycler
    # Interning for `use(...; range = ...)`. `ids` is an IdDict, so two `use`
    # calls naming the same slice would otherwise be two objects and two ids —
    # and a resource that is not the same resource in two passes has no hazards
    # between them, which is the one answer that must not be reachable by
    # accident. Keyed by value, so the same slice is the same sub-resource.
    views::Dict{Tuple{Int,UnitRange{Int}},Any}
end

Mantle.Graph(dev::LavaDevice) =
    LavaGraph(dev, Pass[], LavaSurface[], Transient[],
              Dict{Int,Transient}(), Dict{Int,Any}(), IdDict{Any,Int}(), Any[], Recycler(),
              Dict{Tuple{Int,UnitRange{Int}},Any}())

function Mantle.Transient.Buffer(g::LavaGraph, ::Type{T}, n::Integer) where {T}
    t = TransientBuffer{T}(Int(n), typemax(Int), 0, nothing)
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
imageusage(::Type) = Lava.COLOR_USAGE
# The same three as COLOR_USAGE, with the attachment bit that matches the aspect:
# a depth target is worth copying out (a test that asserts the depth buffer beats
# one that asserts its effect on colour) and worth sampling (depth of field,
# ambient occlusion, anything that reads the z it just wrote).
imageusage(::Type{Float32}) = Lava.Vulkan.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                              Lava.Vulkan.IMAGE_USAGE_TRANSFER_SRC_BIT |
                              Lava.Vulkan.IMAGE_USAGE_SAMPLED_BIT

aspect(::Type) = Lava.Vulkan.IMAGE_ASPECT_COLOR_BIT
aspect(::Type{Float32}) = Lava.Vulkan.IMAGE_ASPECT_DEPTH_BIT

# The same question asked of a resource. A swapchain image and a framebuffer are
# colour by construction; only a transient carries an element type to ask.
aspect(::Any) = Lava.Vulkan.IMAGE_ASPECT_COLOR_BIT

"""Which attachment slot a target fills, asked of the target and not of the
argument position, so the two orders of `screen => Clear(c), z => Clear(1f0)`
mean the same thing."""
isdepth(x) = aspect(x) == Lava.Vulkan.IMAGE_ASPECT_DEPTH_BIT

"""
Every usage bit a format is asked for, checked against what the device says the
format can do, before an image is created with it.

`vkCreateImage` fails with `ERROR_FORMAT_NOT_SUPPORTED` and no indication of
which bit was the problem, and the answer is per format and per driver — depth
formats are exactly where that bites, since `D32_SFLOAT` and `D24_UNORM_S8` are
not both available everywhere. This turns it into a message that names the format
and the bit.
"""
function checkusage(ctx, fmt::Lava.Vulkan.Format, usage::Lava.Vulkan.ImageUsageFlag)
    props = Lava.Vulkan.get_physical_device_format_properties(ctx.physical_device, fmt)
    have = props.optimal_tiling_features
    VK = Lava.Vulkan
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
function Mantle.Transient.Image(g::LavaGraph, ::Type{T}, size::Tuple{Integer,Integer};
                                srgb::Bool = false, source = nothing,
                                usage::Lava.Vulkan.ImageUsageFlag = imageusage(T)) where {T}
    ctx = g.dev.ctx
    width, height = size
    fmt = Mantle.vkformat(Mantle.Vulkan(), T; srgb)
    checkusage(ctx, fmt, usage)
    img = Lava.image_2d(ctx, width, height, fmt, usage)
    t = TransientImage{T}(Int(width), Int(height), fmt, usage, img,
                          Lava.image_requirements(ctx, img),
                          typemax(Int), 0, nothing, nothing, source)
    push!(g.transients, t)
    t
end

Mantle.Transient.Image(g::LavaGraph, ::Type{T}, source; kw...) where {T} =
    Mantle.Transient.Image(g, T, extent_size(source); source, kw...)

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
    ctx = Lava.vk_context()
    t.width, t.height = w, h
    t.image = Lava.image_2d(ctx, w, h, t.format, t.usage)
    t.req = Lava.image_requirements(ctx, t.image)
    t.view = nothing
    t.memory = nothing
    return true
end

refit!(::Transient) = false

"""
Record that this pass touches `t`.

The interval recorded here is in declaration order and is provisional: Liveness
recomputes it from the scheduled order, because reordering passes is exactly what
changes a transient's lifetime.
"""
function touch!(g::LavaGraph, t::Transient)
    i = length(g.passes)
    t.first = min(t.first, i)
    t.last = max(t.last, i)
    g.transient_by_id[resourceid(g, t)] = t
    t
end
touch!(g::LavaGraph, x) = (resourceid(g, x); x)

resourceid(g::LavaGraph, r) = get!(g.ids, r) do
    id = length(g.ids) + 1
    g.by_id[id] = r
    id
end

"""
What kind of resource an id names, so barrier tracking starts in the right state
machine. Images have layouts and buffers do not, and a buffer tracked as an image
would emit layout transitions for a resource that has none.
"""
resourcekind(::Any) = BufferKind()
resourcekind(::TransientImage) = ImageKind()
resourcekind(::LavaSurface) = ImageKind()
resourcekind(::Lava.LavaFramebuffer) = ImageKind()

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
        load = a isa Pair ? last(a) : Mantle.Keep
        load isa Mantle.LoadOp ||
            throw(ArgumentError("a render target takes Clear(value), Keep or Discard, got $load"))
        touch!(g, tgt)
        if isdepth(tgt)
            p.depth === nothing ||
                throw(ArgumentError("a pass takes one depth target, and this one was given two"))
            p.depth, p.depth_load = tgt, load
            # Tested and written, and no stencil aspect: `NoAccess` is what says
            # the image has none, which is what picks the combined layout over the
            # separate-aspect ones.
            push!(p.usages, resourceid(g, tgt) => Depth{ReadWrite,NoAccess,Mantle.discards(load)})
        else
            push!(p.targets, tgt)
            push!(p.loads, load)
            push!(p.usages, resourceid(g, tgt) =>
                  (Mantle.discards(load) ? ColorAttachment{true} : ColorAttachment{false}))
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
struct Attr{T,R} <: Mantle.Resource
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
drawover(p::PassHandle, n::LavaBuffer{Lava.DrawIndirectCommand}) = indirectcount!(p, n)
drawover(p::PassHandle, n::TransientBuffer{Lava.DrawIndirectCommand}) = indirectcount!(p, n)

function indirectcount!(p::PassHandle, n)
    push!(p.pass.usages, resourceid(p.graph, n) => Mantle.Indirect)
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

Mantle.storage(v::BufferRange) = Mantle.storage(v.parent)
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
    use(pass, x; read, write, range = nothing)

The ordinary case: this pass reads or writes this resource. Named usages survive
only where the role can be picked wrongly.

`range` narrows the claim to a slice, in elements. Two passes that name disjoint
slices of one buffer get no barrier between them, and one that does name a slice
gets a barrier scoped to exactly those bytes.
"""
function Mantle.use(p::PassHandle, x; read::Bool = false, write::Bool = false,
                    range::Union{Nothing,UnitRange{Int}} = nothing)
    read || write || throw(ArgumentError("use() needs read, write, or both"))
    U = Storage{BufferKind, Access{read, write}}
    r = range === nothing ? x : slice(p.graph, x, range)
    push!(p.pass.usages, resourceid(p.graph, r) => U)
    # The parent is what the kernel gets, and what liveness has to see touched.
    touch!(p.graph, x)
end

struct Dispatch
    kernel::Any
    args::Tuple
    ndrange::Any
    group::Any
end

# Whether a dispatch was given a workgroup size is a type, not a branch: the
# launch is per pass per frame and this keeps the call site one expression.
kernelfor(k, ::Nothing) = k(Lava.LavaBackend())
kernelfor(k, group) = k(Lava.LavaBackend(), group)

function Mantle.compute!(f, g::LavaGraph, name::AbstractString)
    p = Pass(name, :compute)
    push!(g.passes, p)
    f(PassHandle(g, p))
    p
end

dispatch!(p::PassHandle, kernel, args, ndrange; group = nothing) =
    push!(p.pass.dispatches, Dispatch(kernel, args, ndrange, group))

# The body goes in `dispatches` beside the `Dispatch`es rather than in a field of
# its own: the compile walks that vector and this is one more thing it can find
# there, so `Pass` does not grow a field only one kind ever sets.
function Mantle.custom!(f, g::LavaGraph, name::AbstractString)
    p = Pass(name, :custom)
    push!(g.passes, p)
    body = f(PassHandle(g, p))
    applicable(body) || throw(ArgumentError(
        "custom!: the block has to return a zero-argument callable — it is what " *
        "runs at record time. Declare the uses, then return the work."))
    push!(p.dispatches, body)
    p
end

"""
What `Update` hands back. Calling it stores a reference and nothing else: it runs
on whatever task set the observable, where there is no command buffer to record
into and no guarantee the GPU is done with the resource.

`pending` is swapped atomically because an observable can fire while `run!` is
reading it.
"""
mutable struct UpdateRef
    resource::Any
    range::Union{Nothing,UnitRange{Int}}
    @atomic pending::Any
end

(r::UpdateRef)(data) = (@atomic r.pending = data; nothing)

"""The single pass every `Update` shares, so one pair of barriers covers them all."""
function updates_pass!(g::LavaGraph)
    for p in g.passes
        p.kind === :update && return p
    end
    p = Pass("updates", :update)
    pushfirst!(g.passes, p)
    p
end

function Mantle.Update(g::LavaGraph, buf; range = nothing)
    p = updates_pass!(g)
    touch!(g, buf)
    id = resourceid(g, buf)
    any(u -> u.first == id, p.usages) || push!(p.usages, id => CopyDst)
    r = UpdateRef(buf, range, nothing)
    push!(g.updates, r)
    r
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
write_update!(g::LavaGraph, bq, r::UpdateRef, data::LavaBuffer) =
    write_update!(g, bq, r, Mantle.storage(data))

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
function rename!(g::LavaGraph, bq, dst::LavaBuffer{T}, data::Lava.LavaArray{T,1}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)

    smb, fmb = data.buf[], fresh.buf[]
    Lava.cmd_copy_buffer!(bq, smb, fmb, nbytes;
                          src_off = smb.pool_offset + data.offset,
                          dst_off = fmb.pool_offset + fresh.offset)

    dst.store = fresh
    retire!(g.recycler, old, dst.capacity * sizeof(T),
            Lava.ensure_active_batch!(bq).signal_value)
    nothing
end

function inplace!(bq, dst, data::Lava.LavaArray{T,1}, from::Integer) where {T}
    store = Mantle.storage(dst)
    dmb, smb = store.buf[], data.buf[]
    Lava.cmd_copy_buffer!(bq, smb, dmb, length(data) * sizeof(T);
                          src_off = smb.pool_offset + data.offset,
                          dst_off = dmb.pool_offset + store.offset + (Int(from) - 1) * sizeof(T))
    nothing
end

# A Mantle buffer says the same thing as its store, so it takes the same route.
rename!(g::LavaGraph, bq, dst::LavaBuffer{T}, data::LavaBuffer{T}) where {T} =
    rename!(g, bq, dst, Mantle.storage(data))
inplace!(bq, dst, data::LavaBuffer, from::Integer) =
    inplace!(bq, dst, Mantle.storage(data), from)

"""In-place, inline in the command buffer. Falls back to renaming when the
update is too big for `cmd_update_buffer` to carry."""
function inplace!(bq, dst, data::AbstractVector{T}, from::Integer) where {T}
    store = Mantle.storage(dst)
    mb = store.buf[]
    off = mb.pool_offset + store.offset + (Int(from) - 1) * sizeof(T)
    n = length(data) * sizeof(T)
    if n <= 65536 && n % 4 == 0 && off % 4 == 0
        src = data isa Vector{T} ? data : collect(data)
        GC.@preserve src Lava.Vulkan.cmd_update_buffer(
            Lava.ensure_active_batch!(bq).cmd_buf, mb.buffer,
            UInt64(off), UInt64(n), Ptr{Cvoid}(pointer(src)))
    else
        Lava.upload!(store, data)      # partial and large: rare, and it stalls
    end
    nothing
end

"""
Replace the contents by pointing the resource at a fresh store.

The copy is *recorded*, not executed. `Lava.upload!` would flush the queue to
make its write safe, but nothing reads the fresh store yet, so there is no hazard
to wait for — the frame's own submit carries the copy, and the barrier into the
first pass that reads it is derived from `CopyDst` like any other usage.

The staging buffer is recycled by size for the same reason the stores are: it is
the same size every frame, and a recorded copy reads it later, so it cannot be
handed out again until the GPU is past this frame.
"""
function rename!(g::LavaGraph, bq, dst::LavaBuffer{T}, data::AbstractVector{T}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)
    host = takehost!(g.recycler, bq, nbytes)

    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src Base.unsafe_copyto!(host.mapped_ptr, Ptr{UInt8}(pointer(src)), nbytes)

    fmb = fresh.buf[]
    Lava.cmd_copy_buffer!(bq, host.buffer, fmb, nbytes;
                          dst_off = fmb.pool_offset + fresh.offset)

    dst.store = fresh
    signal = Lava.ensure_active_batch!(bq).signal_value
    retire!(g.recycler, old, dst.capacity * sizeof(T), signal)
    retire!(g.recycler, host, -nbytes, signal)
    nothing
end

function Mantle.copy!(g::LavaGraph, name::AbstractString, dst, src)
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
    compiled::Lava.CompiledGraphicsPipeline
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
    launch::Lava.LaunchPlan
    iter::I                             # Lava.IterPlan for `nd0`
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
    store::Lava.LavaArray{UInt8,1}
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
    store = Lava.LavaArray{UInt8,1}(undef, (stride * ARG_SLOTS,); bq = dev.bq, unified = true)
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
    want > bq.next_timeline && Lava.submit!(bq)
    if Lava.query_timeline(bq) < want
        Lava.wait_semaphores!(bq, Lava.Vulkan.SemaphoreWaitInfo([bq.timeline_sem], [want]))
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
    old::Lava.Vulkan.ImageLayout
    new::Lava.Vulkan.ImageLayout
    src_stage::Lava.Vulkan.PipelineStageFlag2
    src_access::Lava.Vulkan.AccessFlag2
    dst_stage::Lava.Vulkan.PipelineStageFlag2
    dst_access::Lava.Vulkan.AccessFlag2
end

"""
Turn a transition into a barrier, with stages and access ORed over everything it
waits on and the layout taken from the single state it leaves.

This is the point of the exercise: no call site anywhere names a stage.
"""
function ImageBarrier(resource, t::Mantle.Transition)
    be = Mantle.Vulkan()
    ImageBarrier(resource,
        # A discarding destination does not care what was there, so the old
        # layout is UNDEFINED and no contents have to be preserved.
        Mantle.discards(t.to) ? Lava.Vulkan.IMAGE_LAYOUT_UNDEFINED :
                                Mantle.layout(be, t.from, Src()),
        Mantle.layout(be, t.to, Dst()),
        reduce(|, (Mantle.stages(be, u, Src()) for u in t.waits)),
        reduce(|, (Mantle.access(be, u, Src()) for u in t.waits)),
        Mantle.stages(be, t.to, Dst()),
        Mantle.access(be, t.to, Dst()))
end

function emit_barrier!(bq, b::ImageBarrier)
    barrier = Lava.Vulkan._ImageMemoryBarrier2(
        b.old, b.new,
        Lava.Vulkan.QUEUE_FAMILY_IGNORED, Lava.Vulkan.QUEUE_FAMILY_IGNORED,
        target_image(b.resource),
        Lava.Vulkan._ImageSubresourceRange(aspect(b.resource),
                                           UInt32(0), UInt32(1), UInt32(0), UInt32(1));
        src_stage_mask = b.src_stage, src_access_mask = b.src_access,
        dst_stage_mask = b.dst_stage, dst_access_mask = b.dst_access)
    batch = bq.active_batch
    batch === nothing && (batch = Lava.ensure_active_batch!(bq))
    Lava.Vulkan._cmd_pipeline_barrier_2(batch.cmd_buf,
        Lava.Vulkan._DependencyInfo([], [], [barrier]))
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
    pre::Vector{Mantle.Transition}      # what this pass needs before it runs
    barrier::Any                        # one memory barrier covering the rest
end

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
    pool::Lava.Vulkan.QueryPool
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
    info = Lava.Vulkan.QueryPoolCreateInfo(Lava.Vulkan.QUERY_TYPE_TIMESTAMP, UInt32(nslots))
    pool = Lava.Vulkan.unwrap(Lava.Vulkan.create_query_pool(ctx.device, info))
    period = Float64(Lava.Vulkan.get_physical_device_properties(ctx.physical_device).limits.timestamp_period)
    period > 0 || error("this device reports timestampPeriod = 0, so it cannot time passes")
    Profiler(pool, period, nslots, [pp.pass.name for pp in passes],
             [Float64[] for _ in 1:n], [Float64[] for _ in 1:n], false)
end

"""Keep the last `NSAMPLES`, so a plan that runs for an hour does not grow."""
function sample!(ring::Vector{Float64}, x::Real)
    push!(ring, x)
    length(ring) > Mantle.NSAMPLES && popfirst!(ring)
    ring
end

# Mutable because a window resize re-places every transient, and what comes back
# is a different set of slabs and different barriers. The plan is the user's
# handle, so it is updated rather than replaced.
mutable struct LavaPlan <: Mantle.Plan
    graph::LavaGraph
    transitions::Vector{Mantle.Transition}
    passes::Vector{PassPlan}
    pipelines::Set{Any}
    slabs::Vector{Any}                  # one allocation per arena, kept alive here
    peak::Int
    naive::Int
    profiler::Union{Nothing,Profiler}
    alias::Bool                         # kept, so a recompile means what the first one did
    coalesce::Bool
    policy::Mantle.Policy
    args::ArgMemory                     # laid out at compile, written per frame
    # The recording `bake!` took, or `nothing` while the plan records per run.
    # It pins the argument slot it was captured in — `slotbase` is folded into
    # every address the command buffer holds — so a baked plan stops rotating
    # slots and `nextslot!` is not called for it.
    baked::Any
end

"""The backend object behind a resource: what actually gets bound or copied."""
Mantle.storage(x::LavaBuffer) = x.store
Mantle.storage(x::LavaScalar) = x.store
Mantle.storage(a::Attr) = Mantle.storage(a.resource)
Mantle.storage(t::TransientBuffer) = t.view
Mantle.storage(x) = x

# What a draw hands the shader: the device-side array, not the host handle. Doing
# the conversion here rather than through `Adapt` also drops the per-draw pin —
# a plan holds every resource its draws name for as long as it lives, so pinning
# them into each frame's batch was bookkeeping for a lifetime already guaranteed.
todevice(x::Lava.LavaArray) = Lava.LavaDeviceArray(x)
todevice(x) = x
devarg(a::Base.RefValue) = todevice(Mantle.storage(a[]))
devarg(a) = todevice(Mantle.storage(a))

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
replayed, so constructing Vulkan.jl wrapper objects per frame would be ~2.3 kB of
allocation per barrier for a value that never changes.

The low-level `_` form specifically. Vulkan.jl's wrapper converts to the C struct
on every call and allocates 1840 bytes doing it; `_DependencyInfo` through
`_cmd_pipeline_barrier_2` allocates nothing. Measured, not assumed.

`nothing` when the pass waits for nothing, which is the point: two passes over
disjoint resources then have no barrier between them at all.
"""
function build_pass_barrier(g, ts::Vector{Mantle.Transition})
    isempty(ts) && return nothing
    be = Mantle.Vulkan()
    mem = Lava.Vulkan._MemoryBarrier2[]
    spans = @NamedTuple{buf::Any, handle::UInt64, off::UInt64, len::UInt64,
                        ss::Any, sa::Any, ds::Any, da::Any}[]
    for t in ts
        src_stage = reduce(|, (Mantle.stages(be, u, Src()) for u in t.waits))
        src_access = reduce(|, (Mantle.access(be, u, Src()) for u in t.waits))
        dst_stage = Mantle.stages(be, t.to, Dst())
        dst_access = Mantle.access(be, t.to, Dst())
        r = t.resource == 0 ? nothing : get(g.by_id, t.resource, nothing)
        # A resource that can be renamed gets a global barrier instead of one
        # scoped to its buffer. `st.buf[].buffer` below is baked here, at compile
        # time, and `rename!` points the resource at a *different* store — so a
        # scoped barrier would name the store the plan was compiled with and
        # cover none of the memory that is actually read. Widening loses the span
        # scoping for these resources and is the only correct answer: a handle
        # cannot be baked for something that moves.
        st = (r === nothing || renameable(g, r)) ? nothing : Mantle.storage(r)
        if st === nothing
            push!(mem, Lava.Vulkan._MemoryBarrier2(;
                src_stage_mask = src_stage, src_access_mask = src_access,
                dst_stage_mask = dst_stage, dst_access_mask = dst_access))
        else
            off, len = barrierspan(r, st)
            b = st.buf[].buffer
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
    bufs = Lava.Vulkan._BufferMemoryBarrier2[]
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
        push!(bufs, Lava.Vulkan._BufferMemoryBarrier2(
            Lava.Vulkan.QUEUE_FAMILY_IGNORED, Lava.Vulkan.QUEUE_FAMILY_IGNORED,
            s.buf, off, len;
            src_stage_mask = s.ss, src_access_mask = s.sa,
            dst_stage_mask = s.ds, dst_access_mask = s.da))
        i = j
    end
    Lava.Vulkan._DependencyInfo(mem, bufs, [])
end

function emit_pass_barrier!(bq, dep)
    dep === nothing && return false
    batch = bq.active_batch
    batch === nothing && (batch = Lava.ensure_active_batch!(bq))
    Lava.Vulkan._cmd_pipeline_barrier_2(batch.cmd_buf, dep)
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
    shared = false
    for sl in pl.slabs
        p = get(pl.graph.dev.pools, sl.arena, nothing)
        p === nothing && continue
        # `Base.count`: this module owns the bare name — a draw's vertex count —
        # so the predicate form is not reachable through it.
        if Base.count(wr -> wr.value !== nothing, p.tenants) > 1
            shared = true
            break
        end
    end
    shared || return false
    both = Lava.Vulkan.AccessFlag2(Lava.Vulkan.ACCESS_2_MEMORY_READ_BIT) |
           Lava.Vulkan.AccessFlag2(Lava.Vulkan.ACCESS_2_MEMORY_WRITE_BIT)
    dep = Lava.Vulkan._DependencyInfo(
        [Lava.Vulkan._MemoryBarrier2(;
            src_stage_mask = Lava.STAGE2_ALL_COMMANDS, src_access_mask = both,
            dst_stage_mask = Lava.STAGE2_ALL_COMMANDS, dst_access_mask = both)],
        Lava.Vulkan._BufferMemoryBarrier2[], Lava.Vulkan._ImageMemoryBarrier2[])
    emit_pass_barrier!(bq, dep)
end

"""Bytes actually reserved for transients: the max cross section, not the sum."""
Mantle.peakbytes(pl::LavaPlan) = pl.peak

"""What allocating every transient separately would have cost."""
naivebytes(pl::LavaPlan) = pl.naive

"""
The state a compilation carries between phases. Every field is written by exactly
one phase and read by later ones, which is what lets a phase be run alone.
"""
mutable struct Compile
    graph::LavaGraph
    alias::Bool
    coalesce::Bool
    policy::Mantle.Policy
    deps::Vector{Vector{Int}}                         # Dag
    order::Vector{Int}                                # Schedule
    items::Vector{Mantle.Item}                        # Liveness
    slabs::Vector{Any}                                # Place
    slab::Any
    peak::Int
    naive::Int
    offsets::Vector{Int}
    # Aliasing: pass index => the (new, old) transient pairs whose bytes change
    # hands there. The old index is kept because the barrier is derived from what
    # that transient was last doing, not assumed to be everything.
    alias_begins::Dict{Int,Vector{Tuple{Int,Int}}}
    transitions::Vector{Mantle.Transition}            # Barriers
    prepass::IdDict{Pass,Vector{Mantle.Transition}}
    passes::Vector{PassPlan}                          # Pipelines
    pipelines::Set{Any}
end

Compile(g::LavaGraph; alias = true, coalesce = true, policy = Mantle.Overlap()) =
    Compile(g, alias, coalesce, policy, Vector{Int}[], Int[], Mantle.Item[], Any[], nothing,
            0, 0, Int[], Dict{Int,Vector{Tuple{Int,Int}}}(), Mantle.Transition[],
            IdDict{Pass,Vector{Mantle.Transition}}(), PassPlan[], Set{Any}())

"""Passes in execution order, which is the scheduled order once Schedule has run."""
ordered(c::Compile) = isempty(c.order) ? c.graph.passes : c.graph.passes[c.order]

# ── Dag ───────────────────────────────────────────────────────────────────────
writes_it(U) = Mantle.writes(U)

"""
Whether two resource ids can name the same bytes.

Equal ids do. So does a slice against its own parent, and two slices of one
parent whose ranges intersect — those are different ids, and a scheduler told
they are unrelated is free to reorder two passes that write the same memory.
"""
function overlapping(g::LavaGraph, a::Int, b::Int)
    a == b && return true
    ra, rb = get(g.by_id, a, nothing), get(g.by_id, b, nothing)
    (ra isa BufferRange || rb isa BufferRange) || return false
    # `get`, not `resourceid`: this answers a question and must not hand out an
    # id doing it. A slice's parent is always registered first — `use` touches it
    # before `slice` is reached — so the fallback is unreachable, but a query that
    # can grow `by_id` while the compiler is indexing by id is not worth leaving
    # to that invariant holding.
    pa = ra isa BufferRange ? get(g.ids, ra.parent, 0) : a
    pb = rb isa BufferRange ? get(g.ids, rb.parent, 0) : b
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
function Mantle.run!(::Mantle.Dag, c::Compile)
    ps = c.graph.passes
    g = c.graph
    c.deps = [Int[] for _ in ps]
    for j in eachindex(ps), i in 1:(j - 1)
        shared = false
        for (idj, Uj) in ps[j].usages, (idi, Ui) in ps[i].usages
            overlapping(g, idi, idj) || continue
            (writes_it(Ui) || writes_it(Uj)) || continue
            shared = true
            break
        end
        shared && push!(c.deps[j], i)
    end
    c
end

# ── Schedule ──────────────────────────────────────────────────────────────────
# Width of each memory term in the packed score; two of them share the range.
const SCORE_MAX = 0x1fff

"""
Greedy list scheduling under a packed priority score.

Objectives live in disjoint bit ranges of one integer and are OR'd, so comparing
candidates is one integer compare and the policy is chosen by moving a bit range
rather than by writing a different comparator. That trick is the best single idea
in RPS (`rps_dag_schedule.hpp:180-208`).

The memory term prefers a candidate that frees more bytes than it allocates,
counting a transient's first touch as an allocation and its last as a free.
"""
function Mantle.run!(::Mantle.Schedule, c::Compile)
    ps = c.graph.passes
    n = length(ps)
    n == 0 && return c

    remaining = [length(d) for d in c.deps]
    dependents = [Int[] for _ in 1:n]
    for j in 1:n, i in c.deps[j]
        push!(dependents[i], j)
    end

    # How many *passes* touch each transient, so "last use" is known. Counting
    # usages instead would double-count a pass that binds the same resource
    # twice, which makes every candidate tie and the schedule collapse to
    # declaration order.
    ids(p) = unique(first(u) for u in p.usages)
    touches = Dict{Int,Int}()
    for p in ps, id in ids(p)
        touches[id] = get(touches, id, 0) + 1
    end
    seen = Dict{Int,Int}()
    bytesof = Dict{Int,Int}()
    for (id, t) in c.graph.transient_by_id
        bytesof[id] = nbytes(t)
    end

    maxalloc = max(1, maximum(eachindex(ps); init = 0) do i
        sum(id -> get(bytesof, id, 0), ids(ps[i]); init = 0)
    end)
    mshift = Mantle.memory_shift(c.policy)
    oshift = Mantle.order_shift(c.policy)
    order = Int[]
    ready = [i for i in 1:n if remaining[i] == 0]

    while !isempty(ready)
        best, bestscore = 0, -1
        for k in eachindex(ready)
            i = ready[k]
            alloc = 0
            freed = 0
            for id in ids(ps[i])
                b = get(bytesof, id, 0)
                b == 0 && continue
                get(seen, id, 0) == 0 && (alloc += b)
                get(seen, id, 0) + 1 == touches[id] && (freed += b)
            end
            # Two non-negative terms, not their difference. A difference has to
            # be clamped at zero to fit the bit range, and that clamp throws away
            # exactly the distinction being measured: a pass that breaks even and
            # one that allocates two buffers both come out at zero.
            #
            # Scaled by the largest per-pass allocation rather than by a fixed
            # shift. RPS shifts by 16 (`rps_dag_schedule.hpp:351`) because its
            # buffers are megabytes; ours can be any size, and a fixed shift sends
            # every 16 kB buffer to zero and collapses the schedule back to
            # declaration order.
            mem = clamp((SCORE_MAX * (maxalloc - alloc)) ÷ maxalloc, 0, SCORE_MAX) +
                  clamp((SCORE_MAX * freed) ÷ maxalloc, 0, SCORE_MAX)
            # Clamped because the fields are OR'd, not added: a term that spills
            # its 16 bits would silently corrupt the one above it.
            ord = clamp(n - i, 0, 0xffff)                # earlier declaration scores higher
            score = (mem << mshift) | (ord << oshift)
            if score > bestscore
                best, bestscore = k, score
            end
        end
        i = ready[best]
        deleteat!(ready, best)
        push!(order, i)
        for id in ids(ps[i])
            seen[id] = get(seen, id, 0) + 1
        end
        for j in dependents[i]
            remaining[j] -= 1
            remaining[j] == 0 && push!(ready, j)
        end
    end

    length(order) == n || error("scheduling left $(n - length(order)) passes unreachable: cycle in the DAG")
    c.order = order
    c
end

# ── Liveness ──────────────────────────────────────────────────────────────────
"""
Intervals come from which passes touched a resource, recorded by `touch!` as the
graph was built. Liveness is therefore never a second statement that could
disagree with use.

With `alias = false` every transient claims the whole timeline, so none can share
bytes. That is not a tuning knob so much as a bisection tool: it is how the
aliasing hazard was isolated.
"""
function Mantle.run!(::Mantle.Liveness, c::Compile)
    ts = c.graph.transients
    isempty(ts) && return c
    # Recomputed from the scheduled order, not from declaration order: reordering
    # passes is exactly what changes a transient's interval.
    for t in ts
        t.first, t.last = typemax(Int), 0
    end
    for (pos, p) in enumerate(ordered(c)), (id, _) in p.usages
        t = get(c.graph.transient_by_id, id, nothing)
        t === nothing && continue
        t.first = min(t.first, pos)
        t.last = max(t.last, pos)
    end
    # A transient nothing touched has no interval, and the placer's `Span` reports
    # that as an inverted range with `typemax(Int)` in it — a message about the
    # allocator for a mistake in the graph.
    for (i, t) in enumerate(ts)
        t.last == 0 && throw(ArgumentError(
            "transient $i ($(describe(t))) is never used by any pass. A transient's " *
            "interval is derived from use, so one that nothing reads or writes has " *
            "nothing to place. Give it to a pass, or drop the declaration."))
    end
    lastpass = maximum(t -> t.last, ts) + 1
    c.items = [Mantle.Item(string(i),
                           c.alias ? Mantle.Span(t.first, t.last + 1) : Mantle.Span(0, lastpass),
                           nbytes(t); alignment = alignment(t))
               for (i, t) in enumerate(ts)]
    c
end

# ── Place ─────────────────────────────────────────────────────────────────────
"""
A plan's window onto one arena's pool, and the transients placed in it.

Mutable because the pool underneath can be reallocated by a later, larger tenant:
the offsets stay valid — they are this plan's own placement and nothing about
them changed — but `memory` becomes the new allocation and every transient is
materialised into it again. Holding the offsets here rather than only on
`Compile` is what makes that possible without recompiling the plan.
"""
mutable struct Slab
    arena::Any
    memory::Any
    bytes::Int
    indices::Vector{Int}        # into graph.transients
    offsets::Vector{Int}        # parallel to indices
end

"""
What a set of transients requires of the arena backing them.

The two directions are not symmetrical, and getting them the wrong way round is
memory that works until the one image whose type was excluded is bound. Usage
bits union: the arena must permit everything any tenant does with it. Image
memory type bits intersect: the type has to be one every image can bind to.
"""
requirements(::Buffers, ts) =
    (reduce(|, (extrausage(eltype(t)) for t in ts); init = UInt32(0)), typemax(UInt32))
requirements(::Images, ts) =
    (UInt32(0), reduce(&, (UInt32(t.req.type_bits) for t in ts); init = typemax(UInt32)))

"""The one allocation an arena's pool is backed by."""
backing(::Buffers, dev::LavaDevice, bytes::Int, extra::UInt32, ::UInt32) =
    Lava.LavaArray{UInt8,1}(undef, (max(bytes, 1),); extra_usage = extra)
backing(::Images, dev::LavaDevice, bytes::Int, ::UInt32, bits::UInt32) =
    Lava.device_memory(dev.ctx, max(bytes, 1), bits)

"""
Every tenant materialises into the pool's current allocation again.

Called only when the pool actually reallocated. The plans keep their offsets —
the placement inside a plan is unaffected by the pool growing under it — so this
rebinds storage and touches nothing the compiler decided.
"""
function remap!(p::ArenaPool, a)
    live = 0
    for wr in p.tenants
        pl = wr.value
        pl === nothing && continue          # collected: nobody can run it, nothing to remap
        live += 1
        p.tenants[live] = wr
        # A baked plan cannot be remapped: its recording holds the addresses the
        # old allocation had, and nothing can rewrite a command buffer. Silently
        # re-materialising underneath it produces a replay that reads freed
        # storage — deterministic, quiet, and wrong, which is the worst of the
        # three. So this is where it stops, with the order that would have
        # avoided it: build every plan on a device, then bake.
        pl.baked === nothing || throw(ArgumentError(
            "a plan on this device is baked, and another plan needs arena $a grown — " *
            "which would move the baked plan's transients and leave its recording " *
            "pointing at the old allocation. Build every plan that shares a device " *
            "before baking any of them."))
        for sl in pl.slabs
            sl.arena == a || continue
            sl.memory = p.memory
            for (i, off) in zip(sl.indices, sl.offsets)
                materialize!(pl.graph.transients[i], p.memory, off)
            end
        end
    end
    resize!(p.tenants, live)                # prune here, where the list is already walked
    p
end

"""
Reserve `bytes` in the device's pool for this arena, growing it if needed.

The pool is sized to its largest tenant rather than to their total, which is the
whole point: two plans in one process commit the max. Growing reallocates and
remaps, and that is safe here for the same reason `refit!` is — a plan is
compiled outside the frame loop, with nothing of its own in flight.
"""
function reserve!(dev::LavaDevice, a, bytes::Int, ts)
    p = pool!(dev, a)
    extra, bits = requirements(a, ts)
    want_extra = p.extra | extra
    want_bits = p.typebits & bits
    want_bits == 0 && throw(ArgumentError(
        "arena $a has no memory type every image in it can bind to. This plan's " *
        "images and an earlier plan's on this device intersect to nothing, so one " *
        "allocation cannot serve both."))
    want_bytes = max(p.bytes, bytes)
    if p.memory === nothing || want_bytes > p.bytes ||
       want_extra != p.extra || want_bits != p.typebits
        p.memory = backing(a, dev, want_bytes, want_extra, want_bits)
        p.bytes, p.extra, p.typebits = want_bytes, want_extra, want_bits
        remap!(p, a)
    end
    p
end

"""
Register a compiled plan as holding space in the pools it was placed into.

Weakly, so a plan that is dropped does not keep its graph, its pipelines and
every transient it named alive for the life of the device. A collected tenant is
one nothing can run, so skipping it in [`remap!`](@ref) loses nothing.
"""
function tenant!(dev::LavaDevice, pl)
    for sl in pl.slabs
        p = pool!(dev, sl.arena)
        any(wr -> wr.value === pl, p.tenants) || push!(p.tenants, WeakRef(pl))
    end
    pl
end

"""Give a placed transient its storage."""
function materialize!(t::TransientBuffer, slab, offset)
    t.view = Lava.LavaArray{eltype(t),1}(copy(slab.buf), (t.n,); offset)
end

function materialize!(t::TransientImage{T}, slab, offset) where {T}
    t.memory = slab                       # the image outlives the call; the slab must too
    Lava.bind_image!(Lava.vk_context(), t.image, slab, offset)
    t.view = Lava.image_view(Lava.vk_context(), t.image, t.format, aspect(T))
end

"""Bytes the device's other arenas have already committed, so one arena's bound
is what is left rather than the whole device."""
committed(dev::LavaDevice, except) =
    sum((p.bytes for (k, p) in dev.pools if k != except); init = 0)

"""
The largest single allocation this device will make.

A harder limit than the budget and usually a far smaller one — 4 GB against 29 GB
of budget on the machine this was written on — and the one that actually bounds
an arena, because an arena is *one* allocation: the offsets a placement produces
are into a single contiguous range, so it cannot be split across two.

Measured rather than assumed, because the alternative was a reserve fraction
subtracted from the heap, and every such fraction is either too small to prevent
the failure or too large to be justifiable. A plan needing 5 GB in one arena
cannot run on this device however idle the card is, and `checkcapacity` naming
that is worth more than a percentage that happens to catch it here.
"""
function maxalloc(dev::LavaDevice)
    props = Lava.Vulkan.get_physical_device_properties_2(
        dev.ctx.physical_device, Lava.Vulkan.PhysicalDeviceMaintenance3Properties)
    m3 = props.next::Lava.Vulkan.PhysicalDeviceMaintenance3Properties
    Int(m3.max_memory_allocation_size)
end

"""
Assign offsets, one placement per arena.

Peak is summed over arenas because they are separate allocations; the alternative
would be to report the larger and pretend the other is free.
"""
function Mantle.run!(::Mantle.Place, c::Compile)
    isempty(c.items) && return c
    ts = c.graph.transients
    dev = c.graph.dev
    c.offsets = zeros(Int, length(ts))
    c.peak = 0
    c.naive = sum(nbytes, ts)
    for a in unique(map(arena, ts))
        idx = findall(t -> arena(t) == a, ts)
        # Two bounds, and the smaller wins. What this arena may still have — the
        # other arena is a separate allocation and its bytes are already gone —
        # and what the device will hand over in one piece, which on an APU with a
        # 29 GB budget is 4 GB and is therefore usually the binding one.
        avail = min(maxalloc(dev), Mantle.capacity(dev) - committed(dev, a))
        prob = Mantle.Problem(c.items[idx], avail)
        pl = Mantle.checkcapacity(prob, Mantle.place(prob), "arena $a")
        p = reserve!(dev, a, pl.height, ts[idx])
        offs = [pl.offsets[string(i)] for i in idx]
        push!(c.slabs, Slab(a, p.memory, pl.height, idx, offs))
        c.peak += pl.height
        for (i, off) in zip(idx, offs)
            c.offsets[i] = off
            materialize!(ts[i], p.memory, off)
        end
    end
    c.slab = isempty(c.slabs) ? nothing : first(c.slabs).memory
    c
end

# ── Aliasing ──────────────────────────────────────────────────────────────────
"""
Which passes begin a transient that took over another's bytes.

Aliasing creates a hazard between two resources the usage tracker sees as
unrelated: Y's first write lands on memory X was still reading. Nothing in the
per-resource state can know that, so placement has to say so. RPS carries the
same thing as ResourceAliasingInfo with srcDeactivating / dstActivating.

Found by fuzzing: with aliasing on, derived barriers gave a different answer run
to run; with aliasing off, every seed was stable.
"""
function Mantle.run!(::Mantle.Aliasing, c::Compile)
    ts = c.graph.transients
    for (i, y) in enumerate(ts), (j, x) in enumerate(ts)
        i == j && continue
        arena(x) == arena(y) || continue   # offsets in different allocations never overlap
        x.last < y.first || continue
        c.offsets[i] < c.offsets[j] + nbytes(x) &&
            c.offsets[j] < c.offsets[i] + nbytes(y) || continue
        push!(get!(() -> Tuple{Int,Int}[], c.alias_begins, y.first), (i, j))
    end
    c
end

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
function lastuses(states::Dict{Int,Mantle.ResourceState}, slices::Dict{Int,Vector{Int}},
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
function Mantle.run!(::Mantle.Barriers, c::Compile)
    be = Mantle.Vulkan()
    g = c.graph
    states = Dict{Int,Mantle.ResourceState}()
    settled = Dict{Int,Int}()
    last_barrier = 0
    covered_src_stage = covered_dst_stage = Lava.Vulkan.PipelineStageFlag2(0)
    covered_src_access = covered_dst_access = Lava.Vulkan.AccessFlag2(0)
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
    # Converted, not written bare: Vulkan.jl types the `_2_` constants that still
    # fit in 32 bits as the sync1 flag type, and BitMasks refuses to mix the two.
    anystage = Lava.Vulkan.PipelineStageFlag2(Lava.Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    anyread = Lava.Vulkan.AccessFlag2(Lava.Vulkan.ACCESS_2_MEMORY_READ_BIT)
    anywrite = Lava.Vulkan.AccessFlag2(Lava.Vulkan.ACCESS_2_MEMORY_WRITE_BIT)
    every(x) = ~zero(x)
    widen(s::Lava.Vulkan.PipelineStageFlag2) = (s & anystage) != zero(s) ? every(s) : s
    widen(a::Lava.Vulkan.AccessFlag2) =
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
            parent = g.by_id[pid]
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
        r = g.by_id[id]
        u = get(final, id, nothing)
        u === nothing && (u = initial_state(r))
        u === nothing ? Mantle.ResourceState(resourcekind(r)) :
                        Mantle.ResourceState(resourcekind(r), u)
    end

    for (i, p) in enumerate(ordered(c))
        pre = Mantle.Transition[]
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
        handovers = get(c.alias_begins, i, EMPTY_HANDOVER)
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
            push!(needed, Mantle.Transition(0, first(waits), waits, to))
        end

        if !isempty(needed)
            last_barrier = i
            covered_src_stage = reduce(|, (Mantle.stages(be, u, Src()) for t in needed for u in t.waits))
            covered_src_access = reduce(|, (Mantle.access(be, u, Src()) for t in needed for u in t.waits))
            covered_dst_stage = reduce(|, (Mantle.stages(be, t.to, Dst()) for t in needed))
            covered_dst_access = reduce(|, (Mantle.access(be, t.to, Dst()) for t in needed))
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
function Mantle.run!(::Mantle.Pipelines, c::Compile)
    g = c.graph
    argcursor = 0        # every draw's argument block, laid out once
    # A layout change is per-image and cannot be folded into the pass's one memory
    # barrier, so the needed transitions split by resource kind: images become an
    # image barrier each, everything else is ORed into the memory barrier.
    isimage(t) = t.resource != 0 && resourcekind(g.by_id[t.resource]) isa ImageKind
    for p in ordered(c)
        need = c.prepass[p]
        imgs = [ImageBarrier(g.by_id[t.resource], t) for t in need if isimage(t)]
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
        for d in p.draws
            args = map(devarg, d.args)
            vtt = typeof(Lava.convert_args(args))
            ftt = typeof(Lava.convert_args(map(devarg, d.frag_args)))
            vfn, vtt, ffn, ftt = Lava.resolve_shader_pair(d.shader, vtt, ftt)
            # The pass says whether there is a depth attachment, and with dynamic
            # rendering the pipeline has to declare the same thing: a pipeline
            # built for one and drawn into a pass without it is invalid, and the
            # reverse is too.
            shader, compiled = Lava.ensure_compiled_with_shader!(d.shader, vfn, ffn, vtt, ftt;
                                                                color_format = Lava.Vulkan.Format[target_format(t) for t in p.targets],
                                                                depth_format = p.depth === nothing ?
                                                                    Lava.Vulkan.FORMAT_UNDEFINED :
                                                                    target_format(p.depth))
            push!(c.pipelines, compiled.pipeline)
            # The block is laid out to whichever stage's shader reads it.
            # `ensure_compiled_with_shader!` hands back the vertex shader because
            # that is the usual answer; a fullscreen pass whose vertex stage takes
            # nothing and whose fragment stage reads a g-buffer is the other one.
            isempty(d.frag_args) || (shader = Lava.get_or_compile_gfx(ffn, ftt, :fragment))
            packed = isempty(d.frag_args) ? d.args : d.frag_args
            info = shader.push_info
            nbytes = info.arg_buffer_size +
                     Lava.compute_inline_extra_from_byval(info.byval_llvm_sizes)
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
    args = map(devarg, d.args)
    nd = dispatchrange(d.ndrange)
    # `nothing` for the workgroup size, not `d.group`: a group given to
    # `dispatch!` is baked into the kernel's type by `kernelfor`, and KA reads it
    # from there. Passing it here as well made a second, differently keyed
    # iteration plan for the same launch.
    iter = Lava.get_or_build_iter_plan(obj, nd, nothing, dev.ctx)
    tlas = Lava.find_tlas_in_args(args)
    all_args = (obj.f, iter.ka_ctx, args...)
    launch = Lava.launch_plan(dev.bq, obj.f, all_args, iter.ws_3d, tlas !== nothing)
    CompiledDispatch(launch, iter, nd, obj, obj.f, d.args, d.ndrange,
                     tlas !== nothing, argoff, launch.total_size)
end

# An ndrange fixed when the graph was built, or one that is read per frame — the
# same distinction `drawover` makes for a draw's vertex count.
dispatchrange(n::Integer) = n
dispatchrange(t::Tuple) = t
dispatchrange(x) = count(x)

Mantle.Plan(g::LavaGraph; coalesce::Bool = true, alias::Bool = true,
            profile::Bool = false, policy::Mantle.Policy = Mantle.Overlap()) =
    let c = Mantle.compile!(Compile(g; alias, coalesce, policy))
        tenant!(g.dev,
                LavaPlan(g, c.transitions, c.passes, c.pipelines, c.slabs, c.peak, c.naive,
                         profile ? Profiler(g.dev.ctx, c.passes) : nothing,
                         alias, coalesce, policy, ArgMemory(g.dev, c.passes), nothing))
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
    c = Mantle.compile!(Compile(pl.graph; pl.alias, pl.coalesce, pl.policy))
    pl.transitions, pl.passes, pl.pipelines = c.transitions, c.passes, c.pipelines
    pl.slabs, pl.peak, pl.naive = c.slabs, c.peak, c.naive
    # The recompile placed into the device's pools again and produced fresh
    # `Slab`s; the plan has to be a tenant of those, or a later grow remaps the
    # ones it no longer holds and leaves these pointing at the old allocation.
    tenant!(pl.graph.dev, pl)
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
        Lava.Vulkan.get_query_pool_results(
            ctx.device, prof.pool, UInt32(0), UInt32(prof.nslots),
            sizeof(raw), Ptr{Nothing}(pointer(raw)), UInt64(2 * sizeof(UInt64));
            flags = Lava.Vulkan.QUERY_RESULT_64_BIT |
                    Lava.Vulkan.QUERY_RESULT_WITH_AVAILABILITY_BIT)
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
function Mantle.timings(pl::LavaPlan)
    prof = pl.profiler
    prof === nothing &&
        throw(ArgumentError("this plan was not built to be profiled; use Plan(g; profile = true)"))
    collect!(prof, pl.graph.dev.ctx)
    med(x) = isempty(x) ? NaN : (q = sort(x); q[(length(q) + 1) ÷ 2])
    [Mantle.PassTiming(prof.names[i], pl.passes[i].pass.kind,
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
Mantle.baked(pl::LavaPlan) = pl.baked !== nothing

function Mantle.bake!(pl::LavaPlan)
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
    pl.baked = Lava.capture(bq) do
        Lava.concurrent_dispatch_group() do
            record!(pl, bq; derived = true, suppress = true, updates = false)
        end
    end
    pl
end

function run!(pl::LavaPlan; barriers::Symbol = :derived)
    barriers in (:derived, :backend, :both) ||
        throw(ArgumentError("barriers must be :derived, :backend or :both, got $barriers"))
    g = pl.graph
    bq = g.dev.bq
    # Before this frame overwrites them, and without waiting: see `collect!`.
    pl.profiler === nothing || collect!(pl.profiler, g.dev.ctx)
    # Once per frame, here rather than in `isopen`: a predicate that also pumps
    # the event queue is a surprise, and a frame loop that has to remember to
    # poll is a frame loop that stops responding the day someone forgets.
    isempty(g.surfaces) || Lava.GLFW.PollEvents()
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
        Lava.sync_swapchain!(s.win)
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
        Lava.replay!(pl.baked)
        return nothing
    end
    for s in g.surfaces
        Lava.acquire_next_image!(s.win)
    end
    # One slot of the plan's argument memory per frame, reused only once the GPU
    # has passed the frame that last used it.
    nextslot!(pl.args, bq)
    if barriers === :derived
        # The group is a scope rather than a flag, so nothing leaks past here.
        Lava.concurrent_dispatch_group() do
            record!(pl, bq; derived = true, suppress = true)
        end
    else
        record!(pl, bq; derived = barriers === :both, suppress = false)
    end
    # What this frame signals covers everything written into the slot. A split
    # mid-frame makes later batches with higher values, and the last one covers
    # them all, so reading it after recording is right.
    pl.args.signal[pl.args.slot] = Lava.ensure_active_batch!(bq).signal_value
    for s in g.surfaces
        Lava.present_frame!(bq, s.win)
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
    cmd = Lava.ensure_active_batch!(bq).cmd_buf
    Lava.Vulkan.cmd_write_timestamp(cmd, Lava.Vulkan.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                    prof.pool, UInt32(2i - 2))
    t0 = time_ns()
    r = f()
    sample!(prof.host_ns[i], Float64(time_ns() - t0))
    cmd = Lava.ensure_active_batch!(bq).cmd_buf   # a pass may have split the batch
    Lava.Vulkan.cmd_write_timestamp(cmd, Lava.Vulkan.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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
        Lava.Vulkan.cmd_reset_query_pool(Lava.ensure_active_batch!(bq).cmd_buf,
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
    if p.kind === :update && !any(r -> (@atomic r.pending) !== nothing, g.updates)
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
        for d in pp.dispatches
            record_dispatch!(bq, d, am, base)
        end
        return nothing
    elseif p.kind === :custom
        # The barriers this pass declared are already emitted. Open the batch so
        # the body records into this frame's command buffer rather than opening
        # one of its own, then get out of the way.
        Lava.ensure_active_batch!(bq)
        # Two launches inside one body may depend on each other — a two-pass
        # reduction, a split-K matmul, an operator with a scratch buffer — and
        # nothing declared to the graph says so, because the declaration is about
        # what the pass touches and not about how. So the surrounding concurrent
        # group is lifted and Lava's automatic barrier orders the body's own
        # launches, while the *first* of them skips it: that one is the hazard
        # this pass just emitted a derived barrier for.
        Lava.exclusive_dispatch_group() do
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
        for r in g.updates
            data = @atomic r.pending
            data === nothing && continue
            write_update!(g, bq, r, data)
            @atomic r.pending = nothing
        end
        return nothing
    elseif p.kind === :copy
        src, dst = first_target(p), p.dst
        Lava.copy_image_to_buffer!(bq, Mantle.storage(dst), target_image(src),
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
    Lava.vk_begin_pass!(bq,
                        Lava.Vulkan.ImageView[target_view(t) for t in p.targets],
                        Lava.Vulkan.Image[target_image(t) for t in p.targets],
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
    Lava.vk_set_viewport!(bq,
        Lava.Vulkan.Viewport(0f0, 0f0, Float32(ext.width), Float32(ext.height), 0f0, 1f0),
        Lava.Vulkan.Rect2D(Lava.Vulkan.Offset2D(0, 0), ext))
    base = slotbase(am)
    for d in cds
        record_draw!(bq, d, am, base)
    end
    Lava.vk_end_pass!(bq)
    nothing
end

"""
One draw: write its arguments into the plan's slot and record it.

Its own function because a pass's draws are differently parameterised, so the
loop dispatches once per draw and everything inside is concrete — including
`map(devarg, d.args)`, which allocated a boxed tuple per draw per frame while it
was inlined into the loop.

Nothing is allocated and nothing is pinned: the memory belongs to the plan, and
every resource the arguments name is reachable from the plan for as long as it
lives.
"""
function record_draw!(bq, d::CompiledDraw, am::ArgMemory, base::Int)
    off = base + d.argoff
    info = d.shader.push_info
    Lava.pack_args_direct!(bq, am.ptr + off, am.address + off, info.arg_offsets,
                           info.arg_buffer_size, info.byval_llvm_sizes,
                           map(devarg, d.args))
    # No viewport, no scissor, no pin: the pass set the first two once, and the
    # plan holds the pipeline for longer than any frame.
    emit_draw!(bq, d.compiled, d.count, am.address + off)
end

# Where the counts come from, decided once by type rather than per frame by a
# branch. The indirect one records the same command whatever the numbers are,
# which is why nothing has to be read back to record a frame.
emit_draw!(bq, pipe, n::Integer, addr::UInt64) =
    Lava.vk_draw_in_pass!(bq, pipe, n; push_bda = addr, pin = false)
emit_draw!(bq, pipe, x, addr::UInt64) =
    Lava.vk_draw_in_pass!(bq, pipe, count(x); push_bda = addr, pin = false)
emit_draw!(bq, pipe, c::Commands, addr::UInt64) =
    Lava.vk_draw_indirect_in_pass!(bq, pipe, Mantle.storage(c.resource);
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
         Lava.get_or_build_iter_plan(d.obj, nd, nothing, bq.ctx::Lava.VkContext)::I
    it.nblocks == 0 && return nothing
    off = base + d.argoff
    lp = d.launch
    args = map(devarg, d.args)
    Lava.ensure_active_batch!(bq)
    Lava.pack_args_direct!(bq, am.ptr + off, am.address + off, lp.offsets,
                           lp.arg_buffer_size, lp.byval_sizes,
                           (d.kernel, it.ka_ctx, args...))
    Lava.vk_dispatch!(bq, lp.pipeline, am.address + off, it.block_dims,
                      d.tlas ? Lava.find_tlas_in_args(args) : nothing)
    nothing
end

Base.close(::LavaPlan) = nothing
Base.isopen(s::LavaSurface) = isopen(s.win)

end
