# Graph, plan and pass machinery for the Vulkan backend. From Lava it uses only
# the compiler: `lava_compile_gpu_from_job`, `LavaGPUKernel`, the SPIR-V module.

# Transients never outlive a plan; the barrier at the head of every recording
# ([`headbarrier!`](@ref)) orders one plan's reads before the next plan's writes.
pool(d::LavaDevice) = d.pool

function Device(::VulkanAPI; select = nothing, debug::Union{Nothing,DebugConfig} = nothing)
    # Without `select` or `debug`: the process default. Otherwise a new device
    # that is not made the default (see `defaultdevice!`).
    select === nothing && debug === nothing && return vk_context()
    return LavaDevice(; select, debug = something(debug, DebugConfig()))
end

"""
    defaultdevice!(d::LavaDevice) -> d

Make `d` the process default for `Device(VulkanAPI())` and `defaultbackend()`.
The previous default stays alive with its arrays.
"""
defaultdevice!(d::LavaDevice) = (bind_context!(d); d)

build_accel!(f, d::LavaDevice) = build_accel!(f, d.default_bq)


"""
    capacity(device) -> Int

Bytes a placement may use: the `VK_EXT_memory_budget` budget summed over
device-local heaps, which excludes what other processes hold. Without the
extension it is the heap size, an upper bound.
"""
function capacity(dev::LavaDevice)
    heaps = probe_device_memory_budget(dev)
    total = 0
    for h in heaps
        h.device_local || continue
        total += h.budget > 0 ? h.budget : h.size
    end
    total
end

# A device is its own KernelAbstractions backend and launches on its default
# channel; a channel from `allocate_batch_queue!` is one too. Core's `crosswaits!`
# orders buffers shared between channels with timeline-semaphore waits.

h264decoder(dev::LavaDevice, paramnals::AbstractVector{UInt8}; kw...) =
    VideoDecode.H264Decoder(dev, paramnals; kw...)
supports_float64(::LavaDevice) = true
supports_hwtlas(dev::LavaDevice) = dev.ray_query_available

# Lifecycle verbs (`runtime/lifecycle.jl`) over the device's validation ring,
# submission counter and lost flag.

validating(dev::LavaDevice) = dev.validating

validationmessages!(dev::LavaDevice) =
    (msgs = get_validation_messages(dev); empty!(dev.validation.messages); msgs)

# Counts every `vkQueueSubmit2` on any of the device's queues.
submissions(dev::LavaDevice) = dev.diag.flush_counter[]

# Retirement is the lost flag, which `vk_free!` and the other finalizers check
# (see `mark_device_lost!`).
retire!(dev::LavaDevice) = mark_device_lost!(dev)
retired(dev::LavaDevice) = device_lost(dev)

"""
    reset_device!(dev::LavaDevice; debug = dev.debug) -> LavaDevice

Retire `dev` (save its pipeline cache, release its pool, set the lost flag) and
build a device on the same physical device. If `dev` is the process default, the
new device becomes the default.
"""
function reset_device!(dev::LavaDevice; debug::DebugConfig = dev.debug)
    # `select` takes the enumeration index; a new instance enumerates in the
    # same order.
    select = findfirst(pd -> pd.vks == dev.physical_device.vks,
                       unwrap(VK.enumerate_physical_devices(dev.instance)))
    if VK_CONTEXT_REF[] === dev
        reset_device!(; select, debug)
        return vk_context()
    end
    save_pipeline_cache!(dev)
    destroy_pool!(dev)
    mark_device_lost!(dev)
    return Base.invokelatest(LavaDevice; select, debug)::LavaDevice
end

# Bypasses the pool and returns `nothing` on refusal; `try_vk_alloc` discards the
# refused `vkCreateBuffer`'s validation messages.
function tryallocate(dev::LavaDevice, ::Type{T}, dims::Dims{N}) where {T,N}
    buf = try_vk_alloc(dev.default_bq, max(checkedbytes(T,dims),16))
    buf isa AllocFailure && return nothing
    return LavaArray{T,N}(GPUArrays.DataRef(vk_free!, buf), dims)
end

"""
    screenshot(w::VulkanWindow) -> Matrix

The window contents as a `(width, height)` matrix of BGRA byte tuples. The first
index is x, so `permutedims` it before saving it as an image.
"""
screenshot(w::VulkanWindow) = readback_window(w)

# Persistent `Buffer` and `GPURef` are core types over regions of the same pool
# as transients; this file supplies the Vulkan primitives.

extrausage(::Type{DrawIndirectCommand}) = UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT)
# `repeat!`'s per-iteration flags, read by `vkCmdBeginConditionalRenderingEXT`.
extrausage(::Type{Predicate}) = UInt32(VK.BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT)
# Boxes a kernel writes and an acceleration-structure build reads: see
# `build_blas_aabb(ctx, ::LavaArray{AABB,1})`.
extrausage(::Type{AABB}) = AS_INPUT_USAGE

# Conditional rendering is optional; without it `repeat!` must refuse the graph
# rather than run every iteration.
supportspredicate(d::LavaDevice) = d.conditional_rendering_available

# The kernel cache is Lava's; forwarded so callers need not name Lava.
kernelcompiles(::LavaDevice) = compile_stats()
resetkernelcompiles!(::LavaDevice) = reset_compile_stats!()
bufferusage(::LavaDevice, ::Type{T}) where {T} = extrausage(T)

# Vulkan index buffers need `INDEX_BUFFER_BIT` at allocation; core's default, a
# plain `Buffer`, suffices for Metal.
Mantle.indexbuffer(dev::LavaDevice, indices::AbstractVector{UInt32}) =
    alloc_index_buffer(dev.default_bq, indices)

rawalloc(dev::LavaDevice, ::Persistent, bytes::Int, usage) =
    rawalloc(dev, Buffers(), bytes, usage)
constraintof(::LavaDevice, ::Persistent, ts) = UInt32(0)

devicename(d::LavaDevice) = d.device_name

# The timeline value after which a region retired now is free: the newest
# submission, since nothing is recorded and left unsubmitted.
fence(d::LavaDevice) = driver(d.default_bq).next_timeline

# The one completion predicate: `Pool` and plan argument memory ask the device
# rather than compare raw counters.
passed(bq::SubmitChannel{<:VulkanQueue}, v) = query_timeline(bq) >= v

passed(d::LavaDevice, f) = passed(d.default_bq, f)

"""
    waitfor(d::LavaDevice, f) -> Bool

Wait for the timeline to reach `f`. Returns `false` for an `f` past the newest
submission, which nothing will signal; core's `waitfor!` raises the error.
"""
function waitfor(d::LavaDevice, f)
    passed(d, f) && return true
    f > driver(d.default_bq).next_timeline && return false
    wait_timeline!(d.default_bq, UInt64(f))
    checkexceptions!(d)
    return true
end

"""
    deviceview(dev, a::DeviceArray) -> LavaArray

A borrowed `LavaArray` over `a`'s region. Its `DataRef` releaser is a no-op, so
Mantle stays the owner.
"""
deviceview(::LavaDevice, a::DeviceArray{T,N}) where {T,N} =
    LavaArray{T,N}(copy(memoryof(a).ref), size(a); offset = offset(a))

# All a move asks of the backend; core's `notify_move!` does the rest. Typed on
# the block: a method generic in the memory would be ambiguous with core's
# region-address derivation. Image arenas have no addresses and never reach here.
deviceaddress(::LavaDevice, mem::BufferBlock) = UInt64(mem.address)

# A `Vector{T}` goes as it is; anything else is collected into one first.
upload!(d::LavaDevice, a::DeviceArray{T}, first::Integer, data::Vector{T}) where {T} =
    (copyto!(deviceview(d, a), Int(first), data, 1, length(data)); a)
upload!(d::LavaDevice, a::DeviceArray{T}, first::Integer, data::AbstractVector) where {T} =
    upload!(d, a, first, collect(T, data))
download(d::LavaDevice, a::DeviceArray) = Array(deviceview(d, a))
devicecopy!(d::LavaDevice, dst::DeviceArray, src::DeviceArray,
                   n::Integer) =
    (copyto!(deviceview(d, dst), 1, deviceview(d, src), 1, Int(n)); dst)

# Render target accessors for an offscreen framebuffer; the window has its own.

target_view(fb::VulkanFramebuffer) = fb.color_view
target_image(fb::VulkanFramebuffer) = fb.color_image
target_extent(fb::VulkanFramebuffer) = (Int(fb.width), Int(fb.height))
target_format(fb::VulkanFramebuffer) = fb.color_format

loadop(::KeepOp) = VK.ATTACHMENT_LOAD_OP_LOAD
loadop(::DiscardOp) = VK.ATTACHMENT_LOAD_OP_DONT_CARE
loadop(::Clear) = VK.ATTACHMENT_LOAD_OP_CLEAR


# Core's `TransientImage` with this backend's types for its five backend-object
# parameters.
const VulkanTransientImage{T} = TransientImage{T,VK.Format,VK.ImageUsageFlag,VK.Image,
                                               NamedTuple{(:size, :alignment, :type_bits),
                                                          Tuple{Int,Int,UInt32}}}

# What Vulkan needs a storage-buffer binding aligned to, worst case.
alignment(::LavaDevice, ::TransientBuffer) = 256



imageusage(::LavaDevice, ::Type) = COLOR_USAGE
# Depth targets: the depth attachment bit, plus transfer-source and sampled so
# depth can be read back and sampled.
imageusage(::LavaDevice, ::Type{Float32}) = VK.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                            VK.IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK.IMAGE_USAGE_SAMPLED_BIT

# Derived from the portable `isdepth`, not the reverse.
aspect(x) = isdepth(x) ? VK.IMAGE_ASPECT_DEPTH_BIT : VK.IMAGE_ASPECT_COLOR_BIT

"""
    checkusage(ctx, fmt, usage) -> usage

Throw, naming the format and bit, when `fmt` lacks the feature for a requested
usage bit; `vkCreateImage` reports only `ERROR_FORMAT_NOT_SUPPORTED`. Depth
format support differs between drivers.
"""
function checkusage(ctx, fmt::VK.Format, usage::VK.ImageUsageFlag)
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

# The driver half of `Transient.Image`: create the `VkImage` and query its
# memory requirements.

function makeimage(dev::LavaDevice, ::Type{T}, width::Int, height::Int,
                   srgb::Bool, usage::VK.ImageUsageFlag, source) where {T}
    ctx = dev
    fmt = vkformat(VulkanAPI(), T; srgb)
    checkusage(ctx, fmt, usage)
    img = image_2d(ctx, width, height, fmt, usage)
    return VulkanTransientImage{T}(width, height, fmt, usage, img,
                                   image_requirements(ctx, img),
                                   typemax(Int), 0, nothing, nothing, source)
end

function remakeimage!(dev::LavaDevice, t::VulkanTransientImage)
    ctx = dev
    t.image = image_2d(ctx, t.width, t.height, t.format, t.usage)
    t.req = image_requirements(ctx, t.image)
    return t
end
resourcekind(::VulkanFramebuffer) = ImageKind()


Base.length(a::Attr) = length(a.resource)


"""
    storebytes!(e, target, off, p, n)

Record a store of `n` host bytes from `p` into `target` at `off`: inline with
`vkCmdUpdateBuffer` when allowed (at most 64 KiB, 4-byte aligned), otherwise
through submission-owned host-visible scratch. The destination is held until
completion; nothing is submitted early.
"""
function storebytes!(e::Emitter, target::Union{TransientBuffer,DeviceArray},
                     off::Int, p::Ptr{Cvoid}, n::Int)
    n == 0 && return nothing
    block = (target isa DeviceArray ? memoryof(target) : target.block)::BufferBlock
    dst = block.ref[]::VkManagedBuffer
    base = target isa DeviceArray ? offset(target) : target.offset
    dst_off = pool_offset(dst) + base + off
    if n <= 65536 && n % 4 == 0 && dst_off % 4 == 0
        VK.cmd_update_buffer(e.cmd, dst.buffer, UInt64(dst_off), UInt64(n), p)
        hold!(e, dst)
    else
        r = scratch!(e.owner, n, Readback())
        src = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
        unsafe_copyto!(src.mapped_ptr + offset(r), Ptr{UInt8}(p), n)
        cmd_copy_buffer!(e, src, dst, n;
                         src_off = pool_offset(src) + offset(r), dst_off = dst_off)
    end
    return nothing
end


"""
    argbytes(dev, nbytes) -> (region, address, ptr)

A plan's argument bytes: a `Unified` (device-local, host-visible) region the
plan retires, with its device address and mapped host pointer.
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
`LavaArray` a prepare kernel writes and `emit_dispatch!` reads."""
indirectslot(::LavaDevice, region::Region, off::Int) =
    LavaArray{UInt32,1}(copy((memoryof(region)::BufferBlock).ref), (3,);
                        offset = offset(region) + off)


"""
An image barrier with masks and layouts fixed at compile. The image is looked up
at record time, since a window's target is a different swapchain image each frame.
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
    ImageBarrier(resource, t::Transition)

Stages and access ORed over everything `t` waits on, layouts from the states it
leaves and enters.
"""
function ImageBarrier(resource, t::Transition)
    be = VulkanAPI()
    ImageBarrier(resource,
        # A discarding destination starts from UNDEFINED; contents are dropped.
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

Write one barrier into the emitter's command buffer and return whether one was
written. An `ImageBarrier` looks up its image now, a `_DependencyInfo` was built
at compile, and `nothing` writes none, which lets the GPU overlap the passes.
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

function Profiler(ctx, passes)
    n = length(passes)
    nslots = 2n
    info = VK.QueryPoolCreateInfo(VK.QUERY_TYPE_TIMESTAMP, UInt32(nslots))
    pool = VK.unwrap(VK.create_query_pool(ctx.device, info))
    period = Float64(VK.get_physical_device_properties(ctx.physical_device).limits.timestamp_period)
    period > 0 || error("this device reports timestampPeriod = 0, so it cannot time passes")
    Profiler(pool, period, nslots, [pp.pass.name for pp in passes],
             [Float64[] for _ in 1:n], [Float64[] for _ in 1:n], zeros(UInt64, n), false)
end


"""
    storage(t::TransientBuffer, block::BufferBlock) -> LavaArray

A borrowed `LavaArray` over `t`'s bytes in `block`. Its `DataRef` releaser is a
no-op, so Mantle stays the owner. Typed on the block: a
`storage(t::TransientBuffer{T})` method would overwrite core's, for Metal too.
"""
storage(t::TransientBuffer{T,N}, block::BufferBlock) where {T,N} =
    LavaArray{T,N}(copy(block.ref), size(t); offset = t.offset)

# Draw and dispatch arguments reach the shader through the `Adapt` rules, since
# they can be nested structs of arrays (work queues, BVHs). The conversion holds
# nothing; the plan holds every argument it names for its lifetime.


"""
    build_pass_barrier(g, ts) -> Union{Nothing,VK._DependencyInfo}

One global `VkMemoryBarrier2` per distinct hazard from core's
[`barrierhazards`](@ref), naming no buffer handle, since a recording cannot bake
a handle for anything that can move. Built once at compile with VK.jl's
non-allocating `_` structs. `nothing` when the pass waits for nothing, so
independent passes can overlap.
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


# Per arena kind: buffer usage, host mapping, device addresses.
arenabufferconfig(::Buffers) = (
    usage=VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK.BUFFER_USAGE_STORAGE_BUFFER_BIT,
    mapped=false, addressable=true)
arenabufferconfig(::Unified) = (
    usage=arenabufferconfig(Buffers()).usage | VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    mapped=true, addressable=true)
arenabufferconfig(::Readback) = (usage=VK.BufferUsageFlag(0), mapped=true, addressable=false)

arenamemory(ctx, ::Buffers, req) = device_memory(ctx, req.size, req.type_bits)

function arenamemory(ctx, kind::Union{Unified,Readback}, req)
    # Host-visible and coherent are required; the residency preference is not.
    required = VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK.MEMORY_PROPERTY_HOST_COHERENT_BIT
    preferred = kind isa Unified ? VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT : VK.MEMORY_PROPERTY_HOST_CACHED_BIT
    idx = find_memory_type(ctx, req.type_bits, required; preferred_flags=preferred)
    flags = arenabufferconfig(kind).addressable ? VK.MemoryAllocateFlagsInfo(UInt32(0);
        flags=VK.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) : C_NULL
    return VK.DeviceMemory(ctx.device, UInt64(req.size), idx; next=flags)
end

"""
    rawalloc(dev, kind, bytes, usage) -> BufferBlock

One arena buffer with its own memory, mapped once if host-visible and with a
device address if kernels need one. The core block owns reclamation.
"""
function rawalloc(dev::LavaDevice, kind::Union{Buffers,Unified,Readback}, bytes::Int, usage)
    ctx = dev
    cfg = arenabufferconfig(kind)
    u = VK.BufferUsageFlag(usage) | cfg.usage |
        VK.BUFFER_USAGE_TRANSFER_SRC_BIT | VK.BUFFER_USAGE_TRANSFER_DST_BIT
    buf = unbound_buffer(ctx, max(bytes,1), u)
    req = buffer_requirements(ctx, buf)
    mem = arenamemory(ctx, kind, req)
    bind_buffer!(ctx, buf, mem, 0)
    ptr = cfg.mapped ? Ptr{UInt8}(unwrap(VK.map_memory(ctx.device, mem, 0, UInt64(req.size)))) : Ptr{UInt8}(C_NULL)
    addr = cfg.addressable ? UInt64(VK.get_buffer_device_address(ctx.device, VK.BufferDeviceAddressInfo(buf))) : UInt64(0)
    managed = VkManagedBuffer(buf, mem, addr, ptr, Int(req.size), nothing,
                              Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, addr, Int(req.size), ref)
end

rawalloc(dev::LavaDevice, ::Images, bytes::Int, bits) =
    device_memory(dev, max(bytes,1), bits)

constraintof(::LavaDevice, ::Unified, ts) = UInt32(0)
mergeconstraints(::LavaDevice, ::Unified, a::Integer, b::Integer) = a | b

constraintof(::LavaDevice, ::Readback, ts) = UInt32(0)
mergeconstraints(::LavaDevice, ::Readback, a::Integer, b::Integer) = a | b

"""Destroy the block's buffer and memory now. Destructor failures, expected during
a device reset, are logged rather than thrown."""
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

# Called by core (`retire!`/`reclaim!`) once the last submission naming a
# `vk_free!`d buffer has passed.
rawfree(::LavaDevice, buf::VkManagedBuffer) = destroy_buffer!(buf)

# The same for an acceleration structure; here because `LavaDevice` is defined
# after `raytracing/acceleration.jl` is included.
rawfree(::LavaDevice, as::Union{LavaBLAS, LavaTLAS}) = destroy_now!(as)

# The constraint a set of transients shares: the intersection of image memory
# type bits, the union of buffer usage bits.
constraintof(::LavaDevice, ::Images, ts) = reduce(&, (t.req.type_bits for t in ts))
constraintof(::LavaDevice, ::Buffers, ts) =
    reduce(|, (extrausage(eltype(t)) for t in ts))

# A block can host a request whose bits are a subset of its own.
compatible(::LavaDevice, blk::Integer, req::Integer) = (blk & req) == req

# Buffer arenas union usage bits so every tenant's use is permitted; image arenas
# intersect memory type bits, and an empty intersection throws.
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


function materialize!(dev::LavaDevice, t::VulkanTransientImage{T}, slab, offset) where {T}
    # A `VkImage` binds memory once, so after an arena move this makes a new
    # image; the old one's finalizer destroys it.
    t.memory === nothing || remakeimage!(dev, t)
    t.memory = slab                       # the image outlives the call; the slab must too
    # The graph's device, never the process default.
    bind_image!(dev, t.image, slab, offset)
    t.view = image_view(dev, t.image, t.format, aspect(T))
end

"""
    maxalloc(dev) -> Int

`maxMemoryAllocationSize` (`VK_KHR_maintenance3`), capped at the largest
device-local heap because NVIDIA reports no limit.
"""
function maxalloc(dev::LavaDevice)
    props = VK.get_physical_device_properties_2(
        dev.physical_device, VK.PhysicalDeviceMaintenance3Properties)
    m3 = props.next::VK.PhysicalDeviceMaintenance3Properties
    heap = maximum(h -> h.device_local ? h.size : 0, probe_device_memory_budget(dev))
    return Int(min(m3.max_memory_allocation_size, UInt64(heap)))
end

syncbackend(::LavaDevice) = VulkanAPI()

"""
    compiledraw(c, p::Pass, d, argoff) -> CompiledDraw

Compile a draw: the pipeline for its shaders against the pass's attachments, and
the size of its argument block at core's offset `argoff`.
"""
function compiledraw(c::Compile{LavaDevice}, p::Pass, d, argoff::Int)
    native = compile_vulkan_draw(c.graph.dev, d.shader,
        VK.Format[target_format(t) for t in p.targets],
        p.depth === nothing ? VK.FORMAT_UNDEFINED : target_format(p.depth),
        drawargs(d.args), d.frag_args; bindings=d.bindings)
    shader, compiled = native.shader, native.pipeline
    # A rebindable cell is kept as is so a rebind reaches the packer; a plan
    # holding one is never recorded.
    packed = packedargs(d.args, d.args, d.frag_args)
    info = shader.push_info
    nbytes = info.arg_buffer_size + compute_inline_extra_from_byval(info.byval_llvm_sizes, info.arg_buffer_size)
    indices = d.indices === nothing ? nothing : resolve(c.graph.dev, d.indices)
    return CompiledDraw(compiled, shader, packed, d.count, drawviewport(p, d),
                        d.bindings, indices, d.instances, argoff, nbytes)
end

"""
The pass's barriers as this backend emits them: one image barrier per image
transition, since layout changes are per image, and everything else in the
pass's memory barrier.
"""
function passbarriers(c::Compile{LavaDevice}, p::Pass)
    g = c.graph
    need = c.prepass[p]
    isimage(t) = t.resource != 0 && resourcekind(g.ids.by_id[t.resource]) isa ImageKind
    imgs = [ImageBarrier(g.ids.by_id[t.resource], t) for t in need if isimage(t)]
    return imgs, need, build_pass_barrier(g, filter(!isimage, need))
end

"""
    compile_dispatch(c, d::Dispatch, argoff, indirect) -> CompiledDispatch

Resolve once what KA works out per launch: the iteration plan for the ndrange
and the launch plan, which compiles the kernel, for the argument types. Recording
then only packs and emits; this is the only place a Mantle frame compiles a kernel.
"""
function compile_dispatch(c::Compile{LavaDevice}, d::Dispatch, argoff::Int, indirect::Int)
    dev = c.graph.dev
    # A call (no ndrange) compiles nothing; core's `bake` makes it a `Call`.
    iscall(d) && return bake(c, d)
    # A plain KernelInterface function rather than a `@kernel` constructor, by
    # core's `buildskernel`.
    buildskernel(d.kernel, dev) ||
        return kicompile_dispatch(c, dev, d, argoff, indirect)
    obj = kernelfor(d, dev)
    raw = rawargs(d.args)
    args = readonlyargs(c.graph, argument_usage(dev, d.kernel, d.args, d.ndrange, d.group),
                        raw, devargs(adaptor(dev.default_bq), raw))
    nd = dispatchrange(d.ndrange)
    # `callgroup`: `nothing` when `kernelfor` baked the group into the kernel type
    # (passing it again keys a duplicate plan), the group for a `DynamicSize` kernel.
    wgs = callgroup(obj, d.group)
    iter = get_or_build_iter_plan(obj, nd, wgs, dev)
    tlas = find_tlas_in_args(raw)
    all_args = (obj.f, iter.ka_ctx, args...)
    launch = launch_plan(dev.default_bq, obj.f, all_args, iter.ws_3d, tlas !== nothing)
    CompiledDispatch(launch, iter, nd, obj, wgs, obj.f, d.args, d.ndrange,
                     tlas !== nothing, argoff, launch.total_size, indirect)
end

"""
    kicompile_dispatch(compile, dev, dispatch, argoff, indirect) -> CompiledDispatch

Compile a plain-function `KernelInterface` kernel into the same
`CompiledDispatch` as the KA path. With no `KA.Kernel`, `iter.ka_ctx` is
`nothing`, `all_args[1]` is a zero-sized `nothing` that `ka_launch!` drops, and
the extents come from [`ki_launch_extents`](@ref), as for the immediate launch.
"""
function kicompile_dispatch(c::Compile{LavaDevice}, dev::LavaDevice, d::Dispatch,
                            argoff::Int, indirect::Int)
    raw = rawargs(d.args)
    args = readonlyargs(c.graph, argument_usage(dev, d.kernel, d.args, d.ndrange, d.group),
                        raw, devargs(adaptor(dev.default_bq), raw))
    nd = dispatchrange(d.ndrange)
    kern = kikernel(d.kernel, dev, args)
    wg, blocks = ki_launch_extents(kern, nd, d.group)
    iter = IterPlan(nothing, pad_to_3d(dev, blocks), wg, prod(blocks))
    tlas = find_tlas_in_args(raw)
    all_args = (nothing, args...)
    launch = launch_plan(dev.default_bq, d.kernel, all_args, iter.ws_3d, tlas !== nothing)
    CompiledDispatch(launch, iter, nd, kern, nothing, d.kernel, d.args, d.ndrange,
                     tlas !== nothing, argoff, launch.total_size, indirect)
end

"""
    packdraw!(e, d::CompiledDraw)

Pack a draw's arguments into the plan's slot: once at `record!` for a recorded
plan, every run for an unrecorded one. `drawargs` reads a cell's current
contents, so a `rebind!` reaches the next run. A plan holding a cell is not
`recordable`, so `record!` refuses it before this point.
"""
function packdraw!(e::Emitter, d::CompiledDraw)
    am = e.args
    off = d.argoff
    return pack_gfx_args!(e.owner, rawargs(drawargs(d.args)),
                          am.ptr + off, am.address + off, d.shader.push_info)
end

"""
    collect!(prof::Profiler, dev) -> prof

Read the last profiled frame's timestamps without waiting
(`WITH_AVAILABILITY_BIT`; `WAIT_BIT` would stall every frame). An unavailable
frame is a skipped sample.
"""
function collect!(prof::Profiler, dev::LavaDevice)
    # Core sets `pending` per submission, not per emission: a recording emits its
    # timestamps once but writes them every run.
    prof.pending || return prof
    ctx = dev
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
        # RADV reports a reset, unwritten query as available with value 0, where
        # Vulkan says unavailable. A GPU timestamp is never 0, so skip zeros.
        (raw[lo] == 0 || raw[hi] == 0) && continue
        # Skip a frame already sampled whose slots the next frame has not reset.
        raw[lo] == prof.sampled[i] && continue
        prof.sampled[i] = raw[lo]
        ns = elapsed(raw[lo], raw[hi], prof.period_ns)
        ns === nothing || sample!(prof.gpu_ns[i], ns)
    end
    prof
end


# Both timestamps are `BOTTOM_OF_PIPE`, so the interval is the pass's own barrier
# and work; a `TOP_OF_PIPE` start would bill each pass its predecessor's time. The
# query pool is reset at the head of the recording (`emithead!`) and read after
# the run (`collect!`).
passbegin!(pl::Plan, e::Emitter, i::Integer) =
    VK.cmd_write_timestamp(e.cmd, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pl.profiler.pool, UInt32(2i - 2))
function passend!(pl::Plan, e::Emitter, i::Integer, _)
    VK.cmd_write_timestamp(e.cmd, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pl.profiler.pool, UInt32(2i - 1))
    return NaN
end

"""
    emithead!(e::Emitter, plan)

The start of a plan's commands: one global barrier, since earlier work on the
queue may still read pool memory this plan writes and cross-plan hazards are not
derived, then the profiler's query pool reset.
"""
function emithead!(e::Emitter, pl::Plan)
    headbarrier!(e.cmd, e.ctx)
    if pl.profiler !== nothing
        VK.cmd_reset_query_pool(e.cmd, pl.profiler.pool, UInt32(0),
                                UInt32(pl.profiler.nslots))
    end
    return nothing
end

buffer_head_ordered(::LavaDevice)=true

"""
    emitloophead!(e::Emitter, plan, body)

The head of one loop iteration: the plan-head global barrier and, when the
recording writes timestamps, a reset of the body's query slots, which are
written every iteration. `timings` reports a body pass's last iteration.
"""
function emitloophead!(e::Emitter, pl::Plan, body::AbstractUnitRange)
    headbarrier!(e.cmd, e.ctx)
    if pl.stamped
        VK.cmd_reset_query_pool(e.cmd, pl.profiler.pool, UInt32(2 * (first(body) - 1)),
                                UInt32(2 * length(body)))
    end
    return nothing
end


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

"""
    withpredicate(f, e, (pred, i))

Run `f` in a conditional-rendering scope, so the pass's work is discarded unless
predicate `i` is nonzero. Begin and end must be in one command buffer, which an
emitter guarantees; a scope split across submissions hangs the GPU.
"""
function withpredicate(f, e::Emitter, (pred, i)::Tuple{Any,Int})
    # Without the extension the prepare writes zero groups instead, and
    # `repeat!` refuses gated fixed-size work at build.
    e.ctx.conditional_rendering_available || return f()
    arr = storage(pred)
    mb = arr.buf[]::VkManagedBuffer
    VK.cmd_begin_conditional_rendering_ext(e.cmd,
        VK.ConditionalRenderingBeginInfoEXT(mb.buffer,
            UInt64(arr.offset + i * sizeof(Predicate))))
    try
        f()
    finally
        # An open scope would make every later command conditional on this flag.
        VK.cmd_end_conditional_rendering_ext(e.cmd)
    end
end

# In a recording an update pass emits nothing: stores inside the command buffer
# would replay this run's values forever. The run writes them in front of the
# recording (`emitupdates!`); a one-shot writes them where the pass is.
emitupdate!(::Emitter{<:Recording}, ::Plan, ::PassPlan) = nothing
function emitupdate!(e::Emitter{<:OneShot}, pl::Plan, pp::PassPlan)
    anydirty(pl.hostwritten) || return nothing
    emitbarriers!(e, pp)
    landstores!(StoreEmitter(e), pl.hostwritten)
    return nothing
end

function orderstore!(s::StoreEmitter{<:Emitter}, b, i)
    overlappingstore(b,i) || return nothing
    # Overlapping transfer writes need a barrier; command order does not order them.
    memorybarrier!(s.e.cmd,s.e.ctx,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT)
    return nothing
end

function emitcopy!(e::Emitter, ::Plan, pp::PassPlan)
    p = pp.pass
    src, dst = first_target(p), p.dst
    copy_image_to_buffer!(e, storage(dst), target_image(src),
                          size(src)..., target_format(src);
                          aspect = aspect(src), region = p.viewport)
    return nothing
end

"""
    readback_target(dev, t, rest) -> Matrix

Copy a placed render target to the host after a run, in one submission on the
plan's queue: from `rest` (the usage the plan left it in) to `TRANSFER_SRC`, the
copy into a `Readback` region, and back to `rest` for the next run's barriers.
"""
function readback_target(dev::LavaDevice, t::VulkanTransientImage{T}, rest::Type) where {T}
    t.image === nothing &&
        error("this target has not been placed yet; readback needs a compiled plan")
    w, h = size(t)
    bq = batchqueue(dev)
    return readbackimage(dev, bq, T, w, h) do mb, off
        oneshot!(bq; tag = :readback) do e
            emit_barrier!(e, ImageBarrier(t, Transition(0, rest, Type[rest], CopySrc)))
            copyimage!(e.cmd, mb.buffer, off, target_image(t), (0, 0, w, h); aspect = aspect(t))
            emit_barrier!(e, ImageBarrier(t, Transition(0, CopySrc, Type[CopySrc], rest)))
            hold!(e, mb)
        end
    end
end

# One begin/end per pass, with the clear as pass configuration.
# `transition = false`: `emitbarriers!` already set the layouts.
beginrender!(e::Emitter, ::Plan, pp::PassPlan) =
    beginattachments!(e, pp.pass.targets, pp.pass.loads, pp.pass.depth, pp.pass.depth_load;
                      transition = false)
endrender!(e::Emitter, ::Nothing) = (end_pass!(e); nothing)


emitpreparebarrier!(e::Emitter) = (emit_barrier!(e, preparebarrier()); nothing)

"""Invocations per workgroup of the compiled iteration plan, by which core's
prepare divides a device-written count."""
workgroupsize(d::CompiledDispatch) = UInt32(prod(d.iter.ws_3d))

"""
    preparebarrier()

The barrier after a fused prepare: its shader writes made visible to the
indirect read of the workgroup counts and to the following dispatches. A
function, not a `const`: building it calls Vulkan.jl, and the module precompiles
without a driver.
"""
function preparebarrier()
    return VK._DependencyInfo(
        [VK._MemoryBarrier2(;
            src_stage_mask = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
            src_access_mask = VK.AccessFlag2(VK.ACCESS_2_SHADER_WRITE_BIT),
            dst_stage_mask = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) |
                             VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT),
            dst_access_mask = VK.AccessFlag2(VK.ACCESS_2_SHADER_READ_BIT) |
                              VK.AccessFlag2(VK.ACCESS_2_SHADER_WRITE_BIT) |
                              VK.AccessFlag2(VK.ACCESS_2_INDIRECT_COMMAND_READ_BIT))],
        VK._BufferMemoryBarrier2[], VK._ImageMemoryBarrier2[])
end

# Not cached in a global: a backend reads no process-global state, and building
# it costs three small allocations per device-sized dispatch per recording.

# `{L,K,A,I}` follows `CompiledDispatch`'s declaration order: `I` is the
# iteration plan type the lookup below asserts.
"""
    emitdispatch!(e, d::CompiledDispatch, name)

Pack one dispatch's arguments into the plan's memory and emit it. Nothing here
compiles or allocates: an unchanged ndrange reuses the compiled iteration plan,
a moved one costs a shape-keyed lookup.
"""
function emitdispatch!(e::Emitter, d::CompiledDispatch{L,K,A,I},
                       name::AbstractString) where {L,K,A,I}
    nd = dispatchrange(d.ndrange)
    it = nd == d.nd0 ? d.iter :
         get_or_build_iter_plan(d.obj, nd, d.group, e.ctx)::I
    it.nblocks == 0 && return nothing
    am = e.args
    tlas = packdispatch!(e, d, it)
    emitlaunch!(e, d.ndrange, lp_of(d), am.address + d.argoff, it, tlas,
                indirectof(am, d.indirect), name)
    nothing
end



# No `kernelfor` method here: core's `kernelfor(k, ::Nothing, backend)` keeps the
# backend's pinned channel, and a more specific one would build a default backend.
adaptor(e::Emitter) = LavaAdaptor(e.owner)
# At compile there is no owner; the conversion holds nothing.
adaptor(::SubmitChannel{<:VulkanQueue}) = LavaAdaptor(nothing)


# Portable constructors, on the backend's own device rather than the default
# context.
Framebuffer(d::LavaDevice, w::Integer, h::Integer; kw...) = VulkanFramebuffer(w, h; ctx = d, kw...)
# Not sRGB, unlike `VulkanWindow`'s default: a portable window stores what the
# shader wrote, as on Metal.
Window(d::LavaDevice, w::Integer, h::Integer; color_format::Type = BGRA{N0f8},
       srgb::Bool = false, kw...) =
    VulkanWindow(w, h; ctx = d, color_format, srgb, kw...)
Texture2D(d::LavaDevice, data::AbstractArray; kw...) = VulkanTexture2D(data; ctx = d, kw...)
Sampler(d::LavaDevice; kw...) = VulkanSampler(; ctx = d, kw...)

# A `LavaArray`, which is capacity-aware on `resize!`, unlike the generic
# fallback in `memory/array.jl`.
devicearray(d::LavaDevice, data::AbstractArray) = LavaArray(data; bq = d.default_bq)


# The plan's timestamp profiler; core's `Plan` also accepts `nothing`.
makeprofiler(dev::LavaDevice, passes, profile::Bool) =
    profile ? Profiler(dev, passes) : nothing
# Asked once per plan: with `timestampPeriod = 0` there are no usable
# timestamps and building a profiler would throw.
timestamps(dev::LavaDevice) =
    VK.get_physical_device_properties(dev.physical_device).limits.timestamp_period > 0

batchqueue(d::LavaDevice) = d.default_bq


"""
    openrecording(dev::LavaDevice, plan, range) -> Emitter

A command buffer of the plan's own for the recording walk, packing into the
plan's argument memory. `range` is ignored: a Vulkan command buffer grows.
"""
openrecording(dev::LavaDevice, pl::Plan, ::AbstractUnitRange) =
    Emitter(recording!(dev.default_bq), pl.args)

# `runscalls` stays `false` (core's default): a host call would run once at
# record time instead of on every replay, and in an unrecorded walk it would
# submit before the walk's commands, unordered by the derived barriers. Core
# refuses calls at compile.

"""
    closerecording!(e::Emitter, plan) -> Recording

Seal the command buffer and record in the plan's patch table where each packed
device address landed, so a resource that later moves (`resize!`, arena growth)
is patched at submission instead of re-recorded. `seal!` gives the recording the
channel's holds until `release!`, since it is resubmitted every run.
"""
function closerecording!(e::Emitter{<:Recording}, pl::Plan)
    rec = e.owner
    seal!(rec)
    for (addr, at) in rec.patches
        push!(get!(Vector{Tuple{Any,Int}}, pl.patchtab, addr),
              patchtarget(pl, rec, at))
    end
    empty!(rec.patches)
    return rec
end

"""
    patchtarget(plan, rec, at) -> (region, offset)

Where the packed pointer at `at` lives, for later move patches: the plan's
argument memory, or a scratch region the recording owns (`get_arg_buffer`).
Throws for any other memory, which the recording does not own.
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


"""Once per frame: poll events and update the swapchain. `sync_swapchain!`
refuses a closed window, since a destroyed GLFW handle segfaults; a window whose
close button was clicked can still be drawn to."""
function beginframe!(win::VulkanWindow)
    GLFW.PollEvents()
    sync_swapchain!(win)
    return nothing
end

recordsplans(::LavaDevice) = true

"""
    openrun(dev, plan) -> Emitter

A run's one-shot command buffer, opened by core when the run has work in front
of its recording or the plan is windowed (walked per frame). The plan's one
surface must already hold this frame's image, acquired by `beforeframe!`;
nothing here may rebuild the swapchain.
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
    return Emitter(openoneshot(dev.default_bq), pl.args)
end

"""
Submit the run in one submission: the front one-shot, if any, then the recording;
a windowed frame's one-shot with the swapchain semaphores and frame fence, then
the present.
"""
function closerun!(dev::LavaDevice, pl::Plan, e::Union{Nothing,Emitter})
    bq = dev.default_bq
    surfaces = pl.graph.surfaces
    if !isempty(surfaces)
        win = only(surfaces).win
        presentready!(e, win)
        seal!(e.owner)
        return submit_and_present!(bq, win, e.owner)
    end
    # A headless plan that cannot be recorded (`recordable`) has no recording
    # and was walked into `e`.
    return submitplan!(bq, pl, e, Recording)
end

# No recording: a headless plan holding a rebindable cell (see `recordable`)
# submits only the one-shot the walk filled.
submitrecording!(bq, ::Nothing, e::Emitter) = submitrecording!(bq, (), e)
submitrecording!(bq, rec::Recording, e) = submitrecording!(bq, (rec,), e)

"""
Submit the run's front one-shot, if any, then the recording pieces as core groups
them (`submissionbatches`). With a front the submission holds the pieces; without
one the plan owns them. `recs` is kept from `record!`, so a run allocates nothing.
"""
function submitrecording!(bq, recs::Union{Tuple,Vector{Recording}}, e)
    front = e === nothing ? nothing : e.owner::OneShot
    if front !== nothing
        for r in recs
            hold!(bq, r)
        end
    end
    tok = submitlist!(bq, front, recs)
    for r in recs
        r.token = tok
    end
    handover!(bq, tok, front; tag = :run)
    return tok
end

"""Release a recording whose walk threw, possibly unsealed: its command buffer,
regions, holds and descriptor sets go back to the channel."""
abandonrecording!(::LavaDevice, e::Emitter) = (release!(e.owner::Recording); nothing)

"""Seal a failed run's one-shot so it can be reused and release it with its holds."""
function abandonrun!(dev::LavaDevice, e::Emitter)
    o = e.owner::OneShot
    seal!(o)
    release!(queueof(o), o)
    return nothing
end

"""
    present_frame!(dev::LavaDevice, win::VulkanWindow) -> token

Present the acquired image of a frame nothing drew into, transitioning it from
`UNDEFINED` (`presentuntouched!`). A drawn frame is in
`COLOR_ATTACHMENT_OPTIMAL` and is presented by `closerun!`; Vulkan cannot query
an image's layout, so the two cases need separate paths.
"""
function present_frame!(dev::LavaDevice, win::VulkanWindow)
    o = oneshot(dev.default_bq) do e
        presentuntouched!(e, win)
    end
    return submit_and_present!(dev.default_bq, win, o)
end

"""Present the acquired image of a failed frame untouched, through the same
submit-and-present as a drawn frame so the slot's semaphores stay paired and its
token is set. A no-op if nothing is acquired or the device is lost."""
function abandonframe!(dev::LavaDevice, pl::Plan)
    for s in pl.graph.surfaces
        win = s.win
        (win.acquired && !device_lost(dev)) || continue
        o = oneshot(dev.default_bq) do e
            presentuntouched!(e, win)
        end
        submit_and_present!(dev.default_bq, win, o)
    end
    return nothing
end

"""Write `n` bytes from `p` into `r` at `off` inside the command buffer
(`vkCmdUpdateBuffer`), for pointer patches and small stores."""
function emitinline!(e::Emitter, r::Union{Region,DeviceArray}, off::Int, p::Ptr{Cvoid}, n::Int)
    blk = memoryof(r)::BufferBlock
    VK.cmd_update_buffer(e.cmd, blk.buffer, UInt64(offset(r)) + UInt64(off), UInt64(n), p)
    return nothing
end


"""
    emitdraw!(e, ::Nothing, d::CompiledDraw)

Pack one draw's arguments into the plan's slot and emit it. A separate function
so the loop over differently parameterised draws dispatches once per draw. It
allocates and holds nothing: the plan owns the memory, the recording the resources.
"""
function emitdraw!(e::Emitter, ::Nothing, d::CompiledDraw)
    packdraw!(e, d)
    # A draw's own viewport overrides the pass's full-target one, so several axes
    # share a pass.
    d.viewport === nothing || set_viewport!(e, d.viewport...)
    # Through the binding, so a `DrawBinding` draw uses this run's count, index
    # buffer, instances and texture table.
    bind = boundbindings(d.args, d.bindings)
    bind === nothing || use_bindings!(e, d.compiled, bind)
    emit_draw!(e, d.compiled, boundcount(d.args, d.count), e.args.address + d.argoff,
               boundindices(d.args, d.indices), boundinstances(d.args, d.instances))
end

# `emit_draw!` dispatches on the count and index buffer types, fixed when the
# plan is recorded; an indirect count needs no readback. Whether a draw is a
# mesh draw comes from the pipeline, not the count type: `n` is then its
# workgroup count, and a mesh stage has neither instances nor an index buffer.
function emit_draw!(e, pipe, n::Integer, addr::UInt64, ::Nothing, inst::Int)
    ismeshpipeline(pipe) || return draw_in_pass!(e, pipe, n; push_bda = addr, instances = inst)
    inst == 1 || throw(ArgumentError(
        "a mesh draw dispatches workgroups directly; there is no instance count to multiply them by, got $inst"))
    draw_mesh_in_pass!(e, pipe, n; push_bda = addr)
end

emit_draw!(e, pipe, x, addr::UInt64, ::Nothing, inst::Int) =
    emit_draw!(e, pipe, count(x), addr, nothing, inst)

emit_draw!(e, pipe, c::Commands, addr::UInt64, ::Nothing, ::Int) =
    draw_indirect_in_pass!(e, pipe, storage(c.resource);
                           push_bda = addr)

# Held once at record, since the recording names the index buffer for its
# lifetime. The view offset matters: an index buffer may be a pool slice.
# `drawcount` (core) converts the count like a vertex count.
function emit_draw!(e, pipe, n, addr::UInt64, indices, inst::Int)
    ismeshpipeline(pipe) && throw(ArgumentError(
        "a mesh pipeline takes no index buffer: the mesh stage writes its own primitive indices"))
    hold!(e, indices)
    mb = indices.buf[]::VkManagedBuffer
    draw_indexed_in_pass!(e, pipe, drawcount(n); push_bda = addr, instances = inst,
                          indices_buffer = mb.buffer,
                          indices_offset = UInt64(pool_offset(mb) + indices.offset))
end

"""
    packdispatch!(e, d::CompiledDispatch, it) -> tlas

Write one dispatch's arguments into the plan's argument memory and return the
TLAS they name, if any. Done once at `record!`: the command buffer holds the
address of these bytes, so their values are fixed for the plan's life; a value
that changes is a [`GPURef`](@ref), packed as its address.
"""
function packdispatch!(e::Emitter, d::CompiledDispatch, it)
    am = e.args
    off = d.argoff
    lp = d.launch
    raw = rawargs(d.args)
    # Adapt holds nothing, so hold the original leaves here; this also registers
    # their buffers for submission ordering, so a replay cannot race a host write.
    holdleaves!(e.owner, d.kernel)
    holdleaves!(e.owner, raw)
    # Not wrapped by `readonlyargs`: a `Lava.Const` packs as the array it wraps.
    args = devargs(adaptor(e), raw)
    pack_args_direct!(e.owner, am.ptr + off, am.address + off, lp.offsets,
                           lp.arg_buffer_size, lp.byval_sizes,
                           (d.kernel, it.ka_ctx, args...))
    return d.tlas ? find_tlas_in_args(raw) : nothing
end

"""
    emitlaunch!(e, ndrange, lp, argaddr, it, tlas, indirect, name)

Emit the launch: directly for a host-side ndrange, indirectly for a
[`DeviceRange`](@ref) from the workgroup counts the pass's fused prepare
(`emitprepares!`) wrote. The graph orders the count's writer before this
dispatch through its `indirectcount!` `Indirect` read.
"""
emitlaunch!(e::Emitter, ::Any, lp, argaddr, it, tlas, ::Nothing, name) =
    emit_dispatch!(e, lp.pipeline, argaddr, it.block_dims, tlas, name)
emitlaunch!(e::Emitter, ::DeviceRange, lp, argaddr, it, tlas, indirect, name) =
    emit_dispatch!(e, lp.pipeline, argaddr, indirect, tlas, name)

# Graph-declared readback: the transfer and host visibility barrier live in the
# same recording/submission as the producing passes. Fetch only waits and copies.
nativebuffercopy(::LavaDevice) = true

readbackbinding(::LavaDevice, src) = (a = storage(src); (a.buf[], a.offset))
samereadback(a::LavaArray, b::LavaArray) =
    a.buf[] === b.buf[] && a.offset == b.offset && size(a) == size(b)

function emitreadback!(e::Emitter, pl::Plan, pp::PassPlan)
    src = storage(first_target(pp.pass))
    r = pp.pass.dst
    dst = r.buffer.store
    r.recordedsource = readbackbinding(pl.graph.dev, r.source)
    srcbuf = src.buf[]::VkManagedBuffer
    dstbuf = (memoryof(dst)::BufferBlock).ref[]::VkManagedBuffer
    cmd_copy_buffer!(e, srcbuf, dstbuf, sizeof(eltype(r)) * length(r);
                     src_off=pool_offset(srcbuf) + src.offset,
                     dst_off=pool_offset(dstbuf) + offset(dst))
    memorybarrier!(e.cmd, e.ctx, VK_PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_HOST_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK.ACCESS_HOST_READ_BIT)
    return nothing
end

function hostspan(::LavaDevice, block::BufferBlock)
    p = (block.ref[]::VkManagedBuffer).mapped_ptr
    p == C_NULL && throw(ArgumentError("hostspan: buffer is not host mapped"))
    return (p, block.bytes)
end
