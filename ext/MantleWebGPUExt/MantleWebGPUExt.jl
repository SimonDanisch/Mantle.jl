"""
Mantle's WebGPU backend. The GPU is a WebGPU device in a web page (a Bonito
session: an Electron window, or any browser tab); this module decides everything
and the page's executor (`executor.js`) only carries it out.

Memory is a few storage buffers per device, the heaps, each one binding of at most
the device's storage binding size: an array is a byte range of one of them, and a
kernel reaches it through `Lava.HeapPtr`, a heap and an offset (Lava's WebGPU
target). Kernels are compiled by Lava with `TargetFeatures(webgpu = true)`, printed
as WGSL for the device's number of heaps (`Lava.spirv_to_wgsl`) and compiled by the
page.

An extension because Bonito is optional. Every method is written `Mantle.x`/`KI.x`:
a missed qualifier silently defines a new function in this module.
"""
module MantleWebGPUExt

import Adapt
import Bonito
using Bonito: @js_str
import GPUArrays
import KernelAbstractions as KA
import KernelInterface as KI
import Lava
import LinearAlgebra
import Mantle
using Mantle: WebGPUAPI

const EXECUTOR = Bonito.ES6Module(joinpath(@__DIR__, "executor.js"))

# The executor's binding window for one dispatch's arguments (`ARGS_WINDOW` there).
const ARGS_WINDOW = 4096
# The argument buffer: the argument blocks of one submission, each at a multiple of
# `HEAP_ALIGN` with a whole window after it.
const ARGBUFFER = 1 << 20
# Heap ranges start at this alignment (WebGPU's minStorageBufferOffsetAlignment).
const HEAP_ALIGN = 256
# The most bytes one upload message carries (a multiple of 4, so chunks stay whole words).
const UPLOAD_CHUNK = 64 << 20

# ── heap ──────────────────────────────────────────────────────────────────────

"""First-fit allocator of `HEAP_ALIGN`-aligned byte ranges of one heap."""
mutable struct HeapAllocator
    free::Vector{UnitRange{Int}}
end
# The first `HEAP_ALIGN` bytes of every heap are the device's: words 0 and 1 of heap
# 0 are the exception word a kernel that throws raises and the sink its later writes
# go to (Lava's `WEBGPU_EXCEPTION_WORD`, `WEBGPU_SINK_WORD`), and no array starts at
# offset 0, so no heap pointer is the null address.
HeapAllocator(size::Integer) = HeapAllocator([HEAP_ALIGN:Int(size) - 1])

heapbytes(n::Integer) = cld(max(Int(n), 1), HEAP_ALIGN) * HEAP_ALIGN

freebytes(h::HeapAllocator) = sum(length, h.free; init = 0)

"""The offset of a free range of `n` bytes, taken from the heap, or `nothing` if no range is that long."""
function allocate!(h::HeapAllocator, n::Integer)
    n = heapbytes(n)
    for (k, r) in enumerate(h.free)
        length(r) >= n || continue
        rest = first(r) + n:last(r)
        isempty(rest) ? deleteat!(h.free, k) : (h.free[k] = rest)
        return first(r)
    end
    return nothing
end

"""No free range of the heap holds a request; core's pool reclaims and retries on it (`isoutofmemory`)."""
struct HeapExhausted <: Exception
    requested::Int
    free::Int
    ranges::Int
end
Base.showerror(io::IO, e::HeapExhausted) =
    print(io, "WebGPU heap exhausted: $(e.requested) bytes requested, $(e.free) free in $(e.ranges) ranges")
Mantle.isoutofmemory(::HeapExhausted) = true

function release!(h::HeapAllocator, offset::Int, n::Integer)
    push!(h.free, offset:offset + heapbytes(n) - 1)
    sort!(h.free; by = first)
    merged = UnitRange{Int}[]
    for r in h.free
        if !isempty(merged) && last(merged[end]) + 1 == first(r)
            merged[end] = first(merged[end]):last(r)
        else
            push!(merged, r)
        end
    end
    h.free = merged
    return nothing
end

# ── device ────────────────────────────────────────────────────────────────────

"""
    WebGPUDevice

A WebGPU device in a Bonito page. Its heaps hold every array. Launches and
heap writes collect in an open batch, in order, which the next wait submits as one
command buffer (`Mantle.flush!`), a heap read or a plan's run carries in its own
submission. The batch's argument blocks are written to one argument buffer before
it (WebGPU orders a buffer write after the submissions before it, so the batch
still reading the previous arguments sees them). Arrays freed by the GC are only returned to the
heap after the next [`KI.synchronize`](@ref), when no submitted kernel can still be
reading them.

Mantle's pool carves its blocks from the same heaps (`rawalloc`), so a `Buffer`, a
graph's transients and a `KI.allocate`d array are all heap ranges. The timeline
counts submissions; the page reports no per-submission completion, only that all
submitted work is done (`finish`), which a token passes with.
"""
mutable struct WebGPUDevice <: Mantle.Device
    session::Bonito.Session
    id::Int
    heaps::Vector{Int}              # buffer ids in the page: heap k is `heaps[k + 1]`
    args::Int
    heapsize::Int                   # the bytes of each heap
    allocators::Vector{HeapAllocator}
    freed::Vector{Tuple{Int, Int, Int}}  # (heap, offset, bytes) the GC freed, released at the next synchronize
    freedlock::Threads.SpinLock
    info::Dict{Any, Any}            # adapter info, features, limits
    pipelines::Dict{Any, Tuple{Int, Lava.PushConstantInfo}}
    npipelines::Int
    lock::ReentrantLock
    pool::Mantle.Pool
    accesses::Mantle.AccessCache
    # Arrays outside the pool: `KI.allocate`, broadcasts, GPUArrays' results.
    ledger::Mantle.BufferLedger
    # The highest submission (`submissions` below) a `KI.synchronize` has covered.
    completed::UInt64
    # The open batch: its commands, and their argument blocks at the `argoffset`s
    # the commands name.
    batch::Vector{Dict{String, Any}}
    batchargs::Vector{UInt8}
    # Queue submissions the page made for this device (`Mantle.submissions`), and
    # the timeline: a token is the count of submissions that covers its work.
    submissions::Int
    # The last buffer id handed out in the page (the arguments are 1, the heaps 2 on).
    nbuffers::Int
    # The Electron display a `Device(WebGPUAPI())` opened for itself, closed on
    # retirement; `nothing` for a page the caller owns.
    window::Any
    retired::Bool
end

limit(d::WebGPUDevice, name) = Int(d.info["limits"][name])

"""`(heap, offset)` of a range of `n` bytes, from the first heap that has one, or [`HeapExhausted`](@ref)."""
function heaprange!(d::WebGPUDevice, n::Integer)
    lock(d.lock) do
        for (k, h) in enumerate(d.allocators)
            offset = allocate!(h, n)
            offset === nothing || return (k - 1, offset)
        end
        throw(HeapExhausted(heapbytes(n), sum(freebytes, d.allocators),
                            sum(h -> length(h.free), d.allocators)))
    end
end

"""
Call `name(args...)` of the page's executor and wait for its result. A call that
fails in the page throws here with the page's message: the promise is settled
either way, since a rejected one would never reach Julia and the wait would only
time out.
"""
function call(session::Bonito.Session, name::String, args...; timeout = 300.0)
    r = Bonito.evaljs_value(session, js"""$(EXECUTOR).then(M => M[$(name)](...$(collect(Any, args)))).then(
            value => ({ value }), e => ({ error: String(e && e.message || e) }))"""; timeout)
    haskey(r, "error") && error("WebGPU page, $name: $(r["error"])")
    return r["value"]
end
call(d::WebGPUDevice, name::String, args...; kw...) = call(d.session, name, args...; kw...)

"""Send `name(args...)` to the executor without waiting; a failure shows at the next [`call`](@ref) that waits."""
post(d::WebGPUDevice, name::String, args...) =
    Bonito.evaljs(d.session, js"""$(EXECUTOR).then(M => M[$(name)](...$(collect(Any, args))))""")

const WANTED_FEATURES = ["subgroups", "shader-f16", "timestamp-query", "chromium-experimental-subgroup-matrix"]

"""
`Mantle.webgpudevice(page; memory)`: `memory`, 1 GiB by default, is split into as
few heaps as the device's storage binding and buffer sizes allow, at most one fewer
than its storage buffers per shader stage (the arguments take one) and Lava's
`WEBGPU_MAX_HEAPS`. Every kernel binds every heap.
"""
function Mantle.webgpudevice(session::Bonito.Session; power::String = "high-performance",
                             memory::Integer = 2^30)
    info = call(session, "open", 1, power, WANTED_FEATURES)
    limits = info["limits"]
    perheap = min(Int(limits["maxStorageBufferBindingSize"]), Int(limits["maxBufferSize"])) ÷ HEAP_ALIGN * HEAP_ALIGN
    maxheaps = min(Int(limits["maxStorageBuffersPerShaderStage"]) - 1, Lava.WEBGPU_MAX_HEAPS)
    nheaps = cld(Int(memory), perheap)
    nheaps <= maxheaps || throw(ArgumentError(
        "$memory bytes need $nheaps heaps of at most $perheap bytes; this device binds $maxheaps"))
    heapsize = cld(cld(Int(memory), nheaps), HEAP_ALIGN) * HEAP_ALIGN
    heaps = collect(2:nheaps + 1)
    d = WebGPUDevice(session, 1, heaps, 1, heapsize, [HeapAllocator(heapsize) for _ in heaps],
                     Tuple{Int, Int, Int}[], Threads.SpinLock(), info, Dict(), 0, ReentrantLock(),
                     Mantle.Pool(), Mantle.AccessCache(), Mantle.BufferLedger(), UInt64(0),
                     Dict{String, Any}[], UInt8[], 0, nheaps + 1, nothing, false)
    post(d, "createBuffer", d.id, d.args, ARGBUFFER)
    post(d, "createHeaps", d.id, heaps, heapsize)
    return d
end

"""
Take the open batch, its commands and their argument blocks, leaving it empty: for
a submission that carries it (`Mantle.flush!`, a heap read, a run's replay).
"""
function takebatch!(d::WebGPUDevice)
    lock(d.lock) do
        # New vectors: the taken ones become a message.
        taken = (d.batch, d.batchargs)
        d.batch = Dict{String, Any}[]
        d.batchargs = UInt8[]
        return taken
    end
end

"""Count a submission the page makes: the timeline's next token."""
submitted!(d::WebGPUDevice) = (lock(() -> (d.submissions += 1), d.lock); nothing)

"""Submit the open batch as one command buffer, if it holds anything."""
function Mantle.flush!(d::WebGPUDevice)
    commands, args = takebatch!(d)
    isempty(commands) && return nothing
    post(d, "submit", d.id, commands, d.args, args)
    submitted!(d)
    return nothing
end

"""Append a dispatch and its argument block to the open batch, submitting it first if the argument buffer is full."""
function enqueue!(d::WebGPUDevice, cmd::Dict{String, Any}, args::Vector{UInt8})
    lock(d.lock) do
        offset = cld(length(d.batchargs), HEAP_ALIGN) * HEAP_ALIGN
        if offset + ARGS_WINDOW > ARGBUFFER
            Mantle.flush!(d)
            offset = 0
        end
        resize!(d.batchargs, offset)
        append!(d.batchargs, args)
        cmd["argoffset"] = offset
        push!(d.batch, cmd)
    end
    return nothing
end

"""Append a command that takes no arguments (a copy) to the open batch."""
enqueue!(d::WebGPUDevice, cmd::Dict{String, Any}) = (lock(() -> push!(d.batch, cmd), d.lock); nothing)

Mantle.submissions(d::WebGPUDevice) = d.submissions

"""
Chromium switches for WebGPU in an Electron window: the unsafe-WebGPU flag enables the
`chromium-experimental-*` features (subgroup matrix), Vulkan is off by default on
Linux, and the GPU blocklist would hide some Linux drivers.
"""
electron_args() = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist",
                   "--user-data-dir=$(mktempdir())"]

"""
`Mantle.webgpudevice(; power, memory)`: a device on an Electron window of its own,
which needs an Electron backend loaded (`using ElectronCall`) and is closed when
the device is retired.
"""
function Mantle.webgpudevice(; kw...)
    page = Ref{Bonito.Session}()
    app = Bonito.App() do session
        page[] = session
        Bonito.DOM.div("Mantle WebGPU device")
    end
    window = Bonito.HTTPServer.ElectronDisplay(; electron_args = electron_args())
    display(window, app)
    d = Mantle.webgpudevice(page[]; kw...)
    d.window = window
    return d
end

"""
`Device(WebGPUAPI())`: [`Mantle.webgpudevice`](@ref) on a window of its own, with the
default memory. `select` is the adapter's power preference, `"high-performance"`
(the default) or `"low-power"`.
"""
function Mantle.makedevice(::WebGPUAPI, select, debug::Mantle.DebugConfig)
    debug == Mantle.DebugConfig() || throw(ArgumentError(
        "WebGPU: validation is always on in the browser; leave `debug` at its default"))
    return Mantle.webgpudevice(; power = select === nothing ? "high-performance" : String(select))
end

function Mantle.retire!(d::WebGPUDevice)
    d.retired && return nothing
    d.retired = true
    if d.window !== nothing
        close(d.window)
        d.window = nothing
    end
    return nothing
end
Mantle.retired(d::WebGPUDevice) = d.retired

Mantle.devicename(d::WebGPUDevice) = "WebGPU $(d.info["vendor"]) $(d.info["architecture"])"
Mantle.syncbackend(::WebGPUDevice) = WebGPUAPI()
Mantle.waitidle(d::WebGPUDevice) = KI.synchronize(d)

function Mantle.caps(d::WebGPUDevice)
    sub = Int(get(d.info, "subgroupSize", d.info["subgroupMaxSize"]))
    return Mantle.DeviceCaps(false, 0, sub, 0, limit(d, "maxComputeWorkgroupStorageSize"),
                             limit(d, "maxComputeInvocationsPerWorkgroup"), 0, 0)
end

"""
Submit the open batch and wait for every submission, then give the heap back what
the GC freed before it. Throws if a kernel threw since the last synchronize:
WebGPU kernels carry on past a throw (Lava's `continuethrows!`) and only raise the
exception word.
"""
function KI.synchronize(d::WebGPUDevice)
    Mantle.flush!(d)
    freed = takefreed!(d)
    # Read before the wait, so it is credited only with the retirements it covers.
    upto = UInt64(d.submissions)
    threw = call(d, "finish", d.id)
    completed!(d, upto, freed)
    checkexception(d, threw)
    return nothing
end

function takefreed!(d::WebGPUDevice)
    GC.enable_finalizers(false)
    try
        return lock(d.freedlock) do
            r = copy(d.freed)
            empty!(d.freed)
            r
        end
    finally
        GC.enable_finalizers(true)
    end
end

function completed!(d::WebGPUDevice, upto::UInt64, freed)
    lock(d.lock) do
        d.completed = max(d.completed, upto)
        for (heap, offset, n) in freed
            release!(d.allocators[heap + 1], offset, n)
        end
    end
    return nothing
end

function checkexception(d::WebGPUDevice, threw)
    threw == 0 || error("a kernel on $(Mantle.devicename(d)) threw an exception (an out-of-bounds index, " *
                        "a failed conversion, `error`, ...); WebGPU kernels report that one threw, not which or where")
    return nothing
end

KI.max_work_group_size(d::WebGPUDevice) = limit(d, "maxComputeInvocationsPerWorkgroup")
KI.max_work_group_size(k::KI.Kernel{WebGPUDevice}) = KI.max_work_group_size(k.backend)
KI.max_work_group_dims(d::WebGPUDevice) =
    (limit(d, "maxComputeWorkgroupSizeX"), limit(d, "maxComputeWorkgroupSizeY"), limit(d, "maxComputeWorkgroupSizeZ"))
KI.max_num_groups(d::WebGPUDevice) = ntuple(_ -> limit(d, "maxComputeWorkgroupsPerDimension"), 3)

"""
The dispatch grid of a `@kernel`'s `blocks`, split across the axes past the
device's 65535 workgroups per dimension (core's `pad_to_3d`): Lava's lowering of
KernelAbstractions' indices recovers the block from the dispatched grid.
"""
kagrid(d::WebGPUDevice, blocks::Tuple) = Mantle.pad_to_3d(KI.max_num_groups(d), blocks)

"""
`(grid, linear)`: the dispatch grid of a KernelInterface kernel's `groups`, and
whether it is a 1-D grid split across axes, which the kernel is then printed for
(`Lava.spirv_to_wgsl`'s `linear`): such a kernel reads its ids from the builtins.
"""
function kigrid(d::WebGPUDevice, groups::Dims{3})
    maxgroups = KI.max_num_groups(d)
    (groups[1] > maxgroups[1] && groups[2] == 1 && groups[3] == 1) || return groups, false
    return Mantle.pad_to_3d(maxgroups, (groups[1],)), true
end
KI.sub_group_size(d::WebGPUDevice) = Int(d.info["subgroupMaxSize"])

# ── arrays ────────────────────────────────────────────────────────────────────

"""A range of one of a device's heaps: what the `DataRef` of a WebGPU array holds."""
struct HeapBlock
    dev::WebGPUDevice
    heap::Int                       # zero-based, as in `Lava.HeapPtr`
    offset::Int
    nbytes::Int
end

"""The page's buffer id of heap `heap`."""
heapbuffer(d::WebGPUDevice, heap::Integer) = d.heaps[heap + 1]

"""
Where `b` starts as a device address: the 64-bit value whose bytes are those of a
`Lava.HeapPtr` to it, heap in the high word. Core's move patches compare and
rewrite these.
"""
address(b::HeapBlock) = UInt64(b.heap) << 32 | UInt64(b.offset)

"""
Give `b` back to the heap at the next `KI.synchronize`, when no submitted kernel can
still be reading it. The `DataRef` calls this when the last array over the block is
freed, possibly from a finalizer.
"""
function free!(b::HeapBlock)
    d = b.dev
    lock(d.freedlock) do
        push!(d.freed, (b.heap, b.offset, b.nbytes))
    end
    Mantle.freed!(d.ledger, b.nbytes)
    return nothing
end

"""
    WebGPUArray{T,N}

An array in a device's heap. Views, reshapes and reinterprets share the block
(`GPUArrays.derive`); `offset` is the bytes from the block's start to the first
element (bytes, as a reinterpret can start inside a wider element).
"""
mutable struct WebGPUArray{T, N} <: GPUArrays.AbstractGPUArray{T, N}
    data::GPUArrays.DataRef{HeapBlock}
    offset::Int
    dims::NTuple{N, Int}

    # `DataRef` releases on `unsafe_free!` only, never through GC: the finalizer is
    # what frees an array nobody references.
    function WebGPUArray{T, N}(data::GPUArrays.DataRef{HeapBlock}, dims::NTuple{N, Int};
                               offset::Integer = 0) where {T, N}
        a = new{T, N}(data, Int(offset), dims)
        finalizer(GPUArrays.unsafe_free!, a)
        return a
    end
end

function WebGPUArray{T}(d::WebGPUDevice, dims::NTuple{N, Int}) where {T, N}
    (isbitstype(T) && !Base.isbitsunion(T)) ||
        throw(ArgumentError("WebGPU arrays hold isbits elements, not $T"))
    n = prod(dims; init = 1) * sizeof(T)
    heap, offset = heaprange!(d, n)
    Mantle.allocated!(d.ledger, heapbytes(n))
    return WebGPUArray{T, N}(GPUArrays.DataRef(free!, HeapBlock(d, heap, offset, heapbytes(n))), dims)
end

"""
A `WebGPUArray` over `dims` elements at byte `offset` of `blk`, a pool block: the
pool owns the bytes, so the array frees nothing. Keep the region alive while using it.
"""
function borrowed(blk::HeapBlock, ::Type{T}, dims::Dims{N}, offset::Int) where {T, N}
    Mantle.checkspan(blk.nbytes, offset, Mantle.checkedbytes(T, dims))
    return WebGPUArray{T, N}(GPUArrays.DataRef(Returns(nothing), blk), dims; offset)
end

device(a::WebGPUArray) = a.data[].dev
"""The heap of `a` (zero-based), and the byte offset of its first element in it."""
heapindex(a::WebGPUArray) = a.data[].heap
heapoffset(a::WebGPUArray) = a.data[].offset + a.offset

Base.size(a::WebGPUArray) = a.dims
Base.sizeof(a::WebGPUArray{T}) where {T} = length(a) * sizeof(T)
KA.get_backend(a::WebGPUArray) = device(a)
Mantle.Device(a::WebGPUArray) = device(a)
Base.similar(a::WebGPUArray, ::Type{T}, dims::Dims{N}) where {T, N} = WebGPUArray{T}(device(a), dims)

GPUArrays.storage(a::WebGPUArray) = a.data
GPUArrays.derive(::Type{T}, a::WebGPUArray, dims::Dims{N}, offset::Int) where {T, N} =
    WebGPUArray{T, N}(copy(a.data), dims; offset = a.offset + offset * sizeof(T))
# The block's address names the allocation, and the offset is into it: addresses
# are one space (heap in the high word), so arrays of different blocks compare by
# range, and arrays of different heaps never overlap.
GPUArrays.memory_location(a::WebGPUArray) = (UInt(address(a.data[])), a.offset)
Mantle.basealignment(a::WebGPUArray) = Mantle.powalign(UInt64(heapoffset(a)))
# What the graph orders accesses by: the block, and the bytes of it an array covers.
Mantle.resource_identity(a::WebGPUArray) = a.data[]
Mantle.bytespan(a::WebGPUArray{T}) where {T} = (a.offset + 1):(a.offset + length(a) * sizeof(T))

KI.allocate(d::WebGPUDevice, ::Type{T}, dims::Tuple; unified::Union{Nothing, Bool} = nothing) where {T} =
    WebGPUArray{T}(d, map(Int, dims))

"""The kernel's view of `a`: its heap, offset and dimensions."""
deviceview(a::WebGPUArray{T, N}) where {T, N} =
    Lava.HeapDeviceArray{T, N}(Lava.HeapPtr{T}(UInt32(heapoffset(a)), UInt32(heapindex(a))), a.dims)

struct WebGPUAdaptor end
Adapt.adapt_storage(::WebGPUAdaptor, a::WebGPUArray) = deviceview(a)
KI.argconvert(::WebGPUDevice, x) = Adapt.adapt(WebGPUAdaptor(), x)
Adapt.adapt_storage(d::WebGPUDevice, x::Array{T, N}) where {T, N} = copyto!(WebGPUArray{T}(d, size(x)), x)

"""The bytes a read of the page returned: a typed array arrives as a MsgPack extension."""
received(x::AbstractVector{UInt8}) = x
received(x) = getfield(x, :data)::Vector{UInt8}

"""
`n` bytes of heap `heap` from byte `byte`. The page reads whole words, so the range
is widened to them and cut back here. Ordered after everything sent before it.
"""
function readheap(d::WebGPUDevice, heap::Int, byte::Int, n::Int; synchronize::Bool = false)
    first = byte & ~3
    last = (byte + n + 3) & ~3
    # The open batch goes with the read, in one submission.
    freed = synchronize ? takefreed!(d) : nothing
    commands, args = takebatch!(d)
    # Credit only submissions counted before the read. Others may be posted
    # while the page waits for the mapping.
    upto = synchronize ? UInt64(d.submissions) : UInt64(0)
    result = call(d, "readBuffer", d.id, heapbuffer(d, heap), first, last - first, commands, d.args, args, synchronize)
    submitted!(d)
    if synchronize
        completed!(d, upto, freed)
        checkexception(d, result["word"])
        result = result["bytes"]
    end
    words = received(result)
    return words[byte - first + 1:byte - first + n]
end

"""
Write `bytes` to heap `heap` at byte `byte`, as a command of the open batch, so it
lands after the launches before it (a `queue.writeBuffer` lands when called). The
page writes whole words, so a range that starts or ends inside a word is widened
to it with the bytes the word holds, read first.
"""
function writeheap!(d::WebGPUDevice, heap::Int, byte::Int, bytes::Vector{UInt8})
    first = byte & ~3
    last = (byte + length(bytes) + 3) & ~3
    if first != byte || last != byte + length(bytes)
        words = Vector{UInt8}(undef, last - first)
        words[1:4] = readheap(d, heap, first, 4)
        last - first > 4 && (words[end - 3:end] = readheap(d, heap, last - 4, 4))
        words[byte - first + 1:byte - first + length(bytes)] = bytes
        bytes = words
    end
    enqueue!(d, Dict{String, Any}("op" => "write", "dst" => heapbuffer(d, heap), "offset" => first, "data" => bytes))
    return nothing
end

# Bytes are copied, not reinterpreted: a struct with padding has no `reinterpret`.
# An upload is a submission of its own, with the launches before it, as on Vulkan:
# left in the open batch, a region freed after it would wait for whatever submits
# next, which an idle application never does. A run's host stores are the
# exception, written into the run's own submission (`WebGPURun`).
# Sent in chunks of at most `UPLOAD_CHUNK` bytes, one submission each: a page message
# holds its whole payload, and a weight tensor can be gigabytes.
function Base.copyto!(dst::WebGPUArray{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer) where {T}
    n == 0 && return dst
    Mantle.copybounds(dst, doffs, src, soffs, n)
    d = device(dst)
    start = heapoffset(dst) + (doffs - 1) * sizeof(T)
    total = n * sizeof(T)
    GC.@preserve src for at in 0:UPLOAD_CHUNK:total - 1
        bytes = Vector{UInt8}(undef, min(UPLOAD_CHUNK, total - at))
        unsafe_copyto!(pointer(bytes), Ptr{UInt8}(pointer(src, soffs)) + at, length(bytes))
        writeheap!(d, heapindex(dst), start + at, bytes)
        Mantle.flush!(d)
    end
    return dst
end

function Base.copyto!(dst::Array{T}, doffs::Integer, src::WebGPUArray{T}, soffs::Integer, n::Integer) where {T}
    n == 0 && return dst
    Mantle.copybounds(dst, doffs, src, soffs, n)
    bytes = readheap(device(src), heapindex(src), heapoffset(src) + (soffs - 1) * sizeof(T), n * sizeof(T))
    GC.@preserve dst bytes unsafe_copyto!(Ptr{UInt8}(pointer(dst, doffs)), pointer(bytes), length(bytes))
    return dst
end

KA.@kernel function copy_kernel!(dst, doffs, src, soffs)
    i = KA.@index(Global, Linear)
    @inbounds dst[doffs + i] = src[soffs + i]
end

# Within the heap: a kernel, as WebGPU copies need two different buffers.
function Base.copyto!(dst::WebGPUArray{T}, doffs::Integer, src::WebGPUArray{T}, soffs::Integer, n::Integer) where {T}
    n == 0 && return dst
    copy_kernel!(device(dst))(dst, doffs - 1, src, soffs - 1; ndrange = n)
    return dst
end

Base.copyto!(dst::WebGPUArray{T}, src::Array{T}) where {T} = copyto!(dst, 1, src, 1, length(src))
Base.copyto!(dst::Array{T}, src::WebGPUArray{T}) where {T} = copyto!(dst, 1, src, 1, length(src))
Base.copyto!(dst::WebGPUArray{T}, src::WebGPUArray{T}) where {T} = copyto!(dst, 1, src, 1, length(src))
Base.copy(a::WebGPUArray) = copyto!(similar(a), a)
Base.Array(a::WebGPUArray{T, N}) where {T, N} = copyto!(Array{T, N}(undef, a.dims), a)

# ── GPUArrays ─────────────────────────────────────────────────────────────────
#
# GPUArrays' generic implementations (broadcast, map, mapreduce) launch
# KernelAbstractions kernels on `KA.get_backend`, so the backend only names its
# broadcast style and how a broadcast result is allocated.

struct WebGPUArrayStyle{N} <: GPUArrays.AbstractGPUArrayStyle{N} end
WebGPUArrayStyle{M}(::Val{N}) where {N, M} = WebGPUArrayStyle{N}()
Base.BroadcastStyle(::Type{<:WebGPUArray{T, N}}) where {T, N} = WebGPUArrayStyle{N}()

"""The device of the first WebGPU array among a broadcast's arguments, at any depth."""
broadcastdevice(a::WebGPUArray) = device(a)
broadcastdevice(bc::Base.Broadcast.Broadcasted) = broadcastdevice(bc.args)
broadcastdevice(x) = nothing
function broadcastdevice(args::Tuple)
    isempty(args) && return nothing
    d = broadcastdevice(first(args))
    return d === nothing ? broadcastdevice(Base.tail(args)) : d
end

function Base.similar(bc::Base.Broadcast.Broadcasted{WebGPUArrayStyle{N}}, ::Type{T}, dims) where {T, N}
    d = broadcastdevice(bc)
    d === nothing && error("a WebGPU broadcast without a WebGPU array among its arguments")
    return WebGPUArray{T}(d, Base.to_shape(dims))
end

"""A `Ref` in a kernel argument: the value, since a `RefValue` is mutable and not isbits."""
struct DeviceRef{T} <: Ref{T}
    x::T
end
Base.getindex(r::DeviceRef) = r.x
Adapt.adapt_structure(to::WebGPUAdaptor, r::Base.RefValue) = DeviceRef(Adapt.adapt(to, r[]))

"""A `Ref` of a type: the type as a parameter, so the kernel argument takes no space."""
struct DeviceRefType{T} <: Ref{DataType} end
Base.getindex(::DeviceRefType{T}) where {T} = T
Adapt.adapt_structure(::WebGPUAdaptor, r::Base.RefValue{<:Union{DataType, Type}}) = DeviceRefType{r[]}()

# Core's, as for a Vulkan array: GPUArrays' generic one flushes its rescaling to zero.
LinearAlgebra.norm(v::WebGPUArray, p::Real = 2) = Mantle.devicenorm(v, p)

KA.@kernel function fill_kernel!(a, x)
    i = KA.@index(Global, Linear)
    @inbounds a[i] = x
end

function Base.fill!(a::WebGPUArray{T}, x) where {T}
    isempty(a) || fill_kernel!(device(a))(a, convert(T, x); ndrange = length(a))
    return a
end

# ── kernels ───────────────────────────────────────────────────────────────────

KI.kernel_function(d::WebGPUDevice, @nospecialize(f), @nospecialize(tt) = Tuple{}; name = nothing, kwargs...) =
    KI.Kernel(d, f)

"""
    pipeline!(d, f, all_args, items; linear = false) -> (pipeline id, argument layout)

The page's pipeline for `f` at the argument types of `all_args` (the function
slot first) and work-group size `items`: compiled by Lava for WebGPU, printed as
WGSL (for a split 1-D grid with `linear`, see [`kigrid`](@ref)), compiled by the
page. A shader the page rejects is an error showing its messages next to the WGSL.
"""
function pipeline!(d::WebGPUDevice, @nospecialize(f), all_args::Tuple, items::Dims{3}; linear::Bool = false)
    tt = Tuple{map(Core.Typeof, Base.tail(all_args))...}
    key = (typeof(f), tt, items, linear)
    hit = lock(() -> get(d.pipelines, key, nothing), d.lock)
    hit === nothing || return hit
    # In the newest world: a plan walked from an older one launches kernels
    # defined since (a generated family `@eval`s its function while declaring).
    compiled = Base.invokelatest(compilewebgpu, f, tt, items)::Lava.LavaGPUKernel
    wgsl = Lava.spirv_to_wgsl(collect(reinterpret(UInt32, compiled.spirv_bytes)); heaps = length(d.heaps), linear)
    id = lock(() -> (d.npipelines += 1), d.lock)
    r = call(d, "createPipeline", d.id, id, wgsl, "main")
    if !r["ok"]
        numbered = join(("$(lpad(k, 4)) | $l" for (k, l) in enumerate(split(wgsl, '\n'))), "\n")
        msgs = join(("line $(m["line"]):$(m["column"]) $(m["type"]): $(m["message"])" for m in r["messages"]), "\n")
        error("the page rejected the WGSL of $(nameof(typeof(f))):\n$msgs\n\n$numbered")
    end
    entry = (id, compiled.push_info)
    lock(() -> (d.pipelines[key] = entry), d.lock)
    return entry
end

compilewebgpu(@nospecialize(f), @nospecialize(tt), items::Dims{3}) =
    Lava.compile_or_lookup(Lava.lava_kernel_job(f, tt; workgroup_size = items,
                                                features = Lava.TargetFeatures(webgpu = true)))

"""The bytes of a dispatch's argument block: what `layout` (Lava's) says it holds."""
argblockbytes(layout::Lava.PushConstantInfo) = cld(layout.arg_buffer_size, 4) * 4

"""
    packargs!(p, site, layout, all_args)

Write the argument block of a dispatch at `p`: each passed argument's bytes at its
slot, field by field (a struct with padding has no `reinterpret` to bytes). With a
`Mantle.PatchSite` every heap pointer packed is noted for core's move patches; an
eager launch passes `nothing`.
"""
function packargs!(p::Ptr{UInt8}, site, layout::Lava.PushConstantInfo, all_args::Tuple)
    argblockbytes(layout) <= ARGS_WINDOW ||
        error("$(argblockbytes(layout)) bytes of arguments; the binding holds $ARGS_WINDOW")
    slots = Mantle.passedslots([Core.Typeof(a) for a in all_args])
    length(slots) == length(layout.arg_layout) ||
        error("$(length(slots)) passed arguments for $(length(layout.arg_layout)) kernel parameters")
    for (k, i) in enumerate(slots)
        offset, size = layout.arg_layout[k]
        x = all_args[i]
        isbits(x) || error("argument $i, a $(typeof(x)), is not isbits: WebGPU kernels take values and heap pointers")
        # Of the type: `sizeof` of an array value is its elements' bytes.
        sizeof(typeof(x)) == size || error("argument $i: $(sizeof(typeof(x))) bytes for a $size-byte parameter")
        Mantle.storebits!(p + offset, x)
        Mantle.notepacked!(site, typeof(x), p + offset)
    end
    return nothing
end

function packargs(layout::Lava.PushConstantInfo, all_args::Tuple)
    bytes = zeros(UInt8, argblockbytes(layout))
    GC.@preserve bytes packargs!(pointer(bytes), nothing, layout, all_args)
    return bytes
end

function KI.launch(k::KI.Kernel{WebGPUDevice}, groups::Dims{3}, items::Dims{3}, args::Tuple; kwargs...)
    isempty(kwargs) || throw(ArgumentError("WebGPU kernels take no launch keywords, got $(keys(kwargs))"))
    d = k.backend
    # A `Buffer` or `GPURef` is the array over its region, as in `dispatch!`.
    args = Mantle.argmap(Mantle.storage, args)
    all_args = (k.kern, map(a -> KI.argconvert(d, a), args)...)
    # Lava lowers KernelAbstractions' indices from a one-dimensional workgroup: the
    # local index is the x id, and the workgroup's shape is KA's iteration space,
    # in the context argument. Plain KernelInterface kernels read all three ids.
    linear = false
    if first(args) isa KA.CompilerMetadata
        items = (prod(items), 1, 1)
        groups = kagrid(d, (prod(groups),))
    else
        groups, linear = kigrid(d, groups)
    end
    pipeline, layout = pipeline!(d, k.kern, all_args, items; linear)
    cmd = Dict{String, Any}("op" => "dispatch", "pipeline" => pipeline, "groups" => collect(groups),
                            "args" => d.args)
    enqueue!(d, cmd, packargs(layout, all_args))
    return nothing
end

# ── Mantle's pool and graph ──────────────────────────────────────────────────
#
# Verbs only, as on ROCm: core decides the schedule, liveness, placement and the
# pool. A plan is walked every run (`recordsplans` is false): each dispatch is
# core's `Launch`, the same `KI.launch` an eager call makes. WebGPU orders the
# dispatches that share a writable buffer itself, so there are no barriers to
# emit (`needs_transition(::WebGPUAPI, …)` is false).

Mantle.pool(d::WebGPUDevice) = d.pool
Mantle.accesscache(d::WebGPUDevice) = d.accesses
Mantle.ledger(d::WebGPUDevice) = d.ledger
Mantle.devicebuffertype(::WebGPUDevice, ::Type{T}, N::Int) where {T} = Lava.HeapDeviceArray{T, N}
Mantle.argtype(d::WebGPUDevice, @nospecialize(y)) = Core.Typeof(KI.argconvert(d, y))
Mantle.isdevicearray(::Lava.HeapDeviceArray) = true
# A heap offset is an address: integers computed from it reach the heap intrinsics.
Mantle.devicepointer(::Type{<:Lava.HeapPtr}) = true

# Lava's interpreter, as on Vulkan: the WebGPU target changes the code Lava emits,
# not the method table a kernel is inferred through.
Mantle.kernelinterpreter(::WebGPUDevice, kernel, tt) = Lava.kernelinterpreter(kernel, tt)
Mantle.kernelinterpreter(::WebGPUDevice, kernel, tt, threads::Integer) =
    Lava.kernelinterpreter(kernel, tt; workgroup_size = (Int(threads), 1, 1))
Mantle.inferring(f, ::WebGPUDevice) = @lock Lava.INFERENCE_LOCK f()

"""
What Lava's heap intrinsics do to the heap word they are handed (`Lava.heapload`
and its kin, `device/heap.jl` in Lava). The word is computed from a `HeapPtr`, so
it leads back to the argument the pointer came from.
"""
function Mantle.intrinsic_usage(::WebGPUAPI, name::Symbol)
    s = String(name)
    (s == "_lava_wgpu_heapload" || s == "_lava_wgpu_atomic_load") && return Mantle.READ
    (s == "_lava_wgpu_heapstore" || s == "_lava_wgpu_atomic_store") && return Mantle.WRITE
    startswith(s, "_lava_wgpu_atomic_") && return Mantle.ATOMIC
    return nothing
end

# ── allocation ──
#
# A pool block is a range of the heap. No usage flags or memory types: every block
# hosts every request.

const HeapKinds = Union{Mantle.Buffers, Mantle.Images, Mantle.Persistent}

Mantle.constraintof(::WebGPUDevice, ::Union{HeapKinds, Mantle.Readback}, ts) = nothing
Mantle.compatible(::WebGPUDevice, blk, req) = true
Mantle.mergeconstraints(::WebGPUDevice, kind, a, b) = nothing

Mantle.rawalloc(d::WebGPUDevice, ::HeapKinds, bytes::Int, constraint) =
    HeapBlock(d, heaprange!(d, bytes)..., heapbytes(bytes))

"""
A readback block: its range of the heap, where core's copy kernel stages each
readback, and a buffer of the page the host can map (`MAP_READ`), which a run
copies the stagings into ([`mirrorcopies`](@ref)). A buffer copy moves whole words
from a word; the kernel takes a source at any alignment.
"""
struct ReadbackBlock
    block::HeapBlock
    mirror::Int
end

"""The heap range of a pool block."""
heapblock(b::HeapBlock) = b
heapblock(b::ReadbackBlock) = b.block

function Mantle.rawalloc(d::WebGPUDevice, ::Mantle.Readback, bytes::Int, constraint)
    blk = HeapBlock(d, heaprange!(d, bytes)..., heapbytes(bytes))
    id = lock(() -> (d.nbuffers += 1), d.lock)
    post(d, "createReadback", d.id, id, blk.nbytes)
    return ReadbackBlock(blk, id)
end
function Mantle.rawfree(d::WebGPUDevice, rb::ReadbackBlock)
    Mantle.rawfree(d, rb.block)
    post(d, "destroyBuffer", d.id, rb.mirror)
    return nothing
end

# A heap offset is what a kernel holds as a pointer (`Lava.HeapPtr`).
Mantle.deviceaddress(::WebGPUDevice, blk::Union{HeapBlock, ReadbackBlock}) = address(heapblock(blk))

"""A `KI.allocate`d array, or `nothing` when no free range of a heap holds it."""
function Mantle.tryallocate(d::WebGPUDevice, ::Type{T}, dims::Dims) where {T}
    n = Mantle.checkedbytes(T, dims)
    lock(() -> any(h -> any(r -> length(r) >= heapbytes(n), h.free), d.allocators), d.lock) || return nothing
    return WebGPUArray{T}(d, dims)
end

# Core frees a block only when no region in it is live and the device is done
# with them, so the range goes back to the heap at once.
Mantle.rawfree(d::WebGPUDevice, blk::HeapBlock) =
    lock(() -> release!(d.allocators[blk.heap + 1], blk.offset, blk.nbytes), d.lock)

"""
The budget the pool's blocks share (core subtracts what they reserve): the heaps,
less what arrays outside the pool (`KI.allocate`) hold.
"""
Mantle.capacity(d::WebGPUDevice) = length(d.heaps) * d.heapsize - d.ledger.bytes[]

"""Bytes no range of any heap is given to."""
heapfree(d::WebGPUDevice) = lock(() -> sum(freebytes, d.allocators), d.lock)
Mantle.maxalloc(d::WebGPUDevice) = d.heapsize - HEAP_ALIGN
# A block per eighth of the heap: the default 64 MiB is a quarter of a default heap.
Mantle.blocksize(d::WebGPUDevice) = d.heapsize ÷ 8

# Every heap is one of the GPU's buffers; together they are what the device may use.
function Mantle.probe_device_memory_budget(d::WebGPUDevice)
    total = length(d.heaps) * d.heapsize
    return [(heap = 0, device_local = true, size = total, budget = total, usage = total - heapfree(d))]
end

# The browser keeps its own shader cache.
Mantle.save_pipeline_cache!(::WebGPUDevice) = nothing

# ── the timeline ──

"""
The token covering everything sent so far: the last submission, or the open batch's
when it holds work, which is one past every submission made (`issued`) until the
batch is submitted.
"""
Mantle.fence(d::WebGPUDevice) = lock(() -> UInt64(d.submissions + !isempty(d.batch)), d.lock)
Mantle.passed(d::WebGPUDevice, f) = UInt64(f) <= d.completed
Mantle.issued(d::WebGPUDevice) = UInt64(d.submissions)
# The page reports only that all submitted work is done.
Mantle.awaittoken!(d::WebGPUDevice, ::UInt64) = KI.synchronize(d)
Mantle.awaitwrites(d::WebGPUDevice) = KI.synchronize(d)

# ── regions as arrays ──

"""A borrowed `WebGPUArray` over `a`'s region: what a launch, a broadcast or a copy uses."""
Mantle.deviceview(::WebGPUDevice, a::Mantle.DeviceArray{T, N}) where {T, N} =
    borrowed(heapblock(Mantle.memoryof(a)), T, size(a), Mantle.offset(a))

"""A placed transient's slice of its arena. Built once; core's `storage(::TransientBuffer)` returns it."""
Mantle.deviceslice(::WebGPUDevice, ::Type{T}, dims::Dims, blk::HeapBlock, offset::Int) where {T} =
    borrowed(blk, T, dims, offset)

# Heap ranges start at `HEAP_ALIGN`, so a transient does too.
Mantle.alignment(::WebGPUDevice, ::Mantle.TransientBuffer) = HEAP_ALIGN

# ── moving bytes ──

function Mantle.upload!(d::WebGPUDevice, a::Mantle.DeviceArray{T}, first::Integer,
                        data::AbstractVector) where {T}
    src = data isa Vector{T} ? data : collect(T, data)
    copyto!(Mantle.deviceview(d, a), Int(first), src, 1, length(src))
    return a
end

# A read of the heap is ordered after every submission before it.
Mantle.download(d::WebGPUDevice, a::Mantle.DeviceArray) = Array(Mantle.deviceview(d, a))

function Mantle.readscalar(d::WebGPUDevice, a::Mantle.DeviceArray{Int32})
    v = Mantle.deviceview(d, a)
    bytes = readheap(d, heapindex(v), heapoffset(v), sizeof(Int32); synchronize = true)
    return only(reinterpret(Int32, bytes))
end

# ── readbacks ──
#
# Core's copy kernel moves a readback's source into its staging, a heap range of a
# `ReadbackBlock`, at any alignment. The run then copies the staging into the
# block's mirror, which the host maps, in the same submission
# (`mirrorreadbacks!`); a fetch only maps it.

"""The copies of a plan's readback stagings into their mirrors, as batch or recording commands."""
function mirrorcopies(d::WebGPUDevice, pl::Mantle.Plan)
    cmds = Dict{String, Any}[]
    for r in pl.graph.readbacks
        store = r.buffer.store
        nbytes = sizeof(store)
        nbytes == 0 && continue
        rb = Mantle.memoryof(store)::ReadbackBlock
        push!(cmds, Dict{String, Any}("op" => "copy", "src" => heapbuffer(d, rb.block.heap),
                                      "srcoffset" => rb.block.offset + Mantle.offset(store),
                                      "dst" => rb.mirror, "dstoffset" => Mantle.offset(store),
                                      "size" => cld(nbytes, 4) * 4))
    end
    return cmds
end

"""A finished readback, mapped from its block's mirror: no submission."""
function Mantle.readbackdata(d::WebGPUDevice, store::Mantle.DeviceArray{T}) where {T}
    out = Array{T}(undef, size(store))
    nbytes = sizeof(store)
    nbytes == 0 && return out
    rb = Mantle.memoryof(store)::ReadbackBlock
    bytes = received(call(d, "readMapped", d.id, rb.mirror, Mantle.offset(store), cld(nbytes, 4) * 4))
    GC.@preserve out bytes unsafe_copyto!(Ptr{UInt8}(pointer(out)), pointer(bytes), nbytes)
    return out
end

function Mantle.devicecopy!(d::WebGPUDevice, dst::Mantle.DeviceArray{T}, src::Mantle.DeviceArray{T},
                            n::Integer) where {T}
    copyto!(Mantle.deviceview(d, dst), 1, Mantle.deviceview(d, src), 1, Int(n))
    return dst
end

# ── what a dispatch compiles to ──

"""
One compiled dispatch: the page's pipeline, its argument layout, and the launch
geometry, worked out once at `Pipelines` time. Its arguments are packed when it is
recorded, into the plan's argument memory at `argoff`, or when a walked plan calls
it. `args` stays the graph's arguments, resolved at each pack, so a pack after a
move sees the new storage. Untyped fields, as Vulkan's `LavaDispatch`: typed on the
arguments, everything that handles one compiled again for every kernel.
"""
struct WebGPUDispatch
    dev::WebGPUDevice
    pipeline::Int
    layout::Lava.PushConstantInfo
    kernel::Any
    # A `@kernel`'s iteration context, the first argument after the function;
    # `nothing` for a plain KernelInterface kernel, which has none.
    ctx::Any
    args::Tuple
    groups::Dims{3}
    items::Dims{3}
    argoff::Int
end

Mantle.argsize(d::WebGPUDispatch) = argblockbytes(d.layout)
Mantle.indirectindex(::WebGPUDispatch) = 0
Mantle.workgroupsize(d::WebGPUDispatch) = UInt32(prod(d.items))

"""The dispatch's arguments as the kernel takes them, the graph's resolved now."""
function launchargs(cd::WebGPUDispatch)
    d = cd.dev
    adapted = Mantle.argwise(a -> KI.argconvert(d, Mantle.resolve(d, a)), cd.args)
    return cd.ctx === nothing ? (cd.kernel, adapted...) : (cd.kernel, cd.ctx, adapted...)
end

dispatchcommand(cd::WebGPUDispatch, args::Int) =
    Dict{String, Any}("op" => "dispatch", "pipeline" => cd.pipeline, "groups" => collect(cd.groups),
                      "args" => args)

# Walked: packed now, into the open batch.
function (cd::WebGPUDispatch)()
    prod(cd.groups) == 0 && return nothing
    enqueue!(cd.dev, dispatchcommand(cd, cd.dev.args), packargs(cd.layout, launchargs(cd)))
    return nothing
end

"""
    compile_dispatch(compile, dispatch, argoff, indirect) -> WebGPUDispatch

The pipeline and launch geometry of `dispatch`, as an eager launch would work them
out: a `@kernel` through its launch shape (one-dimensional workgroups, as
`KI.launch` gives Lava's KernelAbstractions lowering), a plain kernel through
KernelInterface's extents. A call goes to core's `bake`, and so does a
`DeviceRange` without a ceiling, whose count is read on the host: there is no
indirect dispatch here yet, and `recordspass` keeps such a plan walked.
"""
function Mantle.compile_dispatch(c::Mantle.Compile{WebGPUDevice}, d::Mantle.Dispatch,
                                 argoff::Int, indirect::Int)
    dev = c.graph.dev
    Mantle.iscall(d) && return Mantle.bake(c, d)
    d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing && return Mantle.bake(c, d)
    raw = Mantle.argwise(a -> Mantle.resolve(dev, a), d.args)
    adapted = Mantle.argwise(a -> KI.argconvert(dev, a), raw)
    nd = Mantle.dispatchrange(d.ndrange)
    if Mantle.buildskernel(d.kernel, dev)
        obj = Mantle.kernelfor(d, dev)
        group = Mantle.callgroup(obj, d.group)
        shape = Mantle.launchshape(obj, nd, group)
        # A `DynamicSize` kernel launched without a size is partitioned as KA's
        # launch tunes it, before anything is compiled: the shape `launchshape`
        # starts from is the whole ndrange in one workgroup.
        if KA.workgroupsize(obj) <: KA.NDIteration.DynamicSize && group === nothing
            threads = KI.launch_configuration(KI.Kernel(dev, obj.f); nitems = prod(shape.ndrange)).workgroupsize
            iterspace, _ = KA.partition(obj, shape.ndrange, KI.threads_to_workgroupsize(threads, shape.ndrange))
            shape = Mantle.LaunchShape(shape.ndrange, iterspace, KA.mkcontext(obj, shape.ndrange, iterspace))
        end
        f, ctx = obj.f, shape.ctx
        items = (Mantle.groupthreads(shape), 1, 1)
        groups = kagrid(dev, size(KA.blocks(shape.iterspace)))
        linear = false
        all_args = (f, ctx, adapted...)
    else
        kern = Mantle.kikernel(d.kernel, dev, raw)
        items, groups = Mantle.ki_launch_extents(kern, nd, d.group)
        groups, linear = kigrid(dev, groups)
        f, ctx = d.kernel, nothing
        all_args = (f, adapted...)
    end
    maxgroups = limit(dev, "maxComputeWorkgroupsPerDimension")
    all(<=(maxgroups), groups) || throw(ArgumentError(
        "dispatch!: $(groups) workgroups of $(items) invocations; a WebGPU dispatch has at most " *
        "$maxgroups workgroups per dimension"))
    pipeline, layout = pipeline!(dev, f, all_args, items; linear)
    return WebGPUDispatch(dev, pipeline, layout, f, ctx, d.args, groups, items, argoff)
end

# ── recordings ──
#
# A plan's recording is a list of commands the page keeps (`record`) and encodes
# into one command buffer per run (`replay`); a WebGPU command buffer is used once.
# The commands bind the plan's argument memory, a buffer of the page that holds
# every dispatch's argument block, packed once when recording. A `resize!` that
# moves a buffer patches the pointers packed to it (core's `notify_move!`), in the
# host copy and in the page's buffer, in front of the next replay.

"""
    WebGPUArgs

A plan's argument memory: a buffer of the page bound as every recorded dispatch's
arguments (binding 1, at the dispatch's offset), and its host copy, which core packs
and patches through `ArgMemory.ptr`.
"""
struct WebGPUArgs
    dev::WebGPUDevice
    id::Int
    host::Vector{UInt8}
end

"""
`(store, address, ptr)` for a plan's `nbytes` of argument blocks. The page's buffer
has a binding window more, as binding 1 covers `ARGS_WINDOW` bytes from the last
block's offset. Kernels reach it by binding, not address: the address is 0.
"""
function Mantle.argbytes(d::WebGPUDevice, nbytes::Int)
    id = lock(() -> (d.nbuffers += 1), d.lock)
    post(d, "createBuffer", d.id, id, nbytes + ARGS_WINDOW)
    a = WebGPUArgs(d, id, zeros(UInt8, nbytes))
    return a, UInt64(0), pointer(a.host)
end

# The page frees a destroyed buffer once the submissions using it are done.
Mantle.retire!(::Mantle.Pool, d::WebGPUDevice, a::WebGPUArgs) = (post(d, "destroyBuffer", d.id, a.id); nothing)

"""
The emitter of a run: of its front (host stores and patches) in front of a
recording, or of the whole walk of an unrecorded plan. Host stores are writes in
the open batch, which the run's submission carries, so a run is one submission.
"""
struct WebGPURun
    dev::WebGPUDevice
end

Mantle.openrun(d::WebGPUDevice, ::Mantle.Plan) = WebGPURun(d)
Mantle.abandonrun!(::WebGPUDevice, ::WebGPURun) = nothing
Mantle.emitupdate!(e::WebGPURun, pl::Mantle.Plan, ::Mantle.PassPlan) =
    (Mantle.landstores!(Mantle.StoreEmitter(e), pl.hostwritten); nothing)
Mantle.emitdispatch!(::WebGPURun, d, ::AbstractString) = (d(); nothing)
Mantle.emitloophead!(::WebGPURun, ::Mantle.Plan, ::AbstractUnitRange) = nothing

"""`n` host bytes at `p` to byte `offset` of array `a`, as a write of the run's batch."""
function storeheap!(e::WebGPURun, a::WebGPUArray, offset::Int, p::Ptr{Cvoid}, n::Int)
    bytes = Vector{UInt8}(undef, n)
    GC.@preserve bytes unsafe_copyto!(pointer(bytes), convert(Ptr{UInt8}, p), n)
    writeheap!(e.dev, heapindex(a), heapoffset(a) + offset, bytes)
    return nothing
end
Mantle.storebytes!(e::WebGPURun, store::Mantle.DeviceArray, offset::Int, p::Ptr{Cvoid}, n::Int) =
    storeheap!(e, Mantle.deviceview(e.dev, store), offset, p, n)
Mantle.storebytes!(e::WebGPURun, t::Mantle.TransientBuffer, offset::Int, p::Ptr{Cvoid}, n::Int) =
    storeheap!(e, Mantle.storage(t), offset, p, n)

"""A patch of a plan's argument memory: into the host copy and the page's buffer, in front of the replay."""
function Mantle.emitinline!(::WebGPURun, am::Mantle.ArgMemory{WebGPUArgs}, off::Int,
                            p::Ptr{Cvoid}, n::Int)
    unsafe_copyto!(am.ptr + off, convert(Ptr{UInt8}, p), n)
    a = am.store
    post(a.dev, "writeBuffer", a.dev.id, a.id, off, a.host[off + 1:off + n])
    return nothing
end

"""The commands of one recorded piece, collected by the plan's walk."""
struct WebGPURecorder
    plan::Mantle.Plan
    passes::UnitRange{Int}
    commands::Vector{Dict{String, Any}}
end

"""A recorded piece, kept by the page under `id`."""
struct WebGPURecording
    dev::WebGPUDevice
    id::Int
end

Mantle.recordsplans(::WebGPUDevice) = true
# A profiled plan is walked: a replay has no pass walk to time.
Mantle.recordsprofiled(::WebGPUDevice) = false

"""Whether a declared dispatch can be recorded: not a `DeviceRange` without a ceiling, whose count the host reads."""
recordabledispatch(d::Mantle.Dispatch) = !(d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing)
recordabledispatch(d) = true
Mantle.recordspass(::WebGPUDevice, p::Mantle.Pass) =
    (p.kind === :compute || p.kind === :update) && all(recordabledispatch, p.dispatches)

function Mantle.openrecording(d::WebGPUDevice, pl::Mantle.Plan, passes::AbstractUnitRange)
    pl.profile && return nothing
    return WebGPURecorder(pl, first(passes):last(passes), Dict{String, Any}[])
end

"""Pack `cd`'s arguments into the plan's argument memory, noting its heap pointers for move patches."""
function Mantle.emitdispatch!(e::WebGPURecorder, cd::WebGPUDispatch, name::AbstractString)
    prod(cd.groups) == 0 && return nothing
    am = e.plan.args::Mantle.ArgMemory{WebGPUArgs}
    block = am.ptr + cd.argoff
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), block, 0, Mantle.argsize(cd))
    GC.@preserve am packargs!(block, Mantle.PatchSite(e.plan, am, am.ptr), cd.layout, launchargs(cd))
    cmd = dispatchcommand(cd, am.store.id)
    cmd["argoffset"] = cd.argoff
    cmd["name"] = String(name)
    push!(e.commands, cmd)
    return nothing
end

# WebGPU orders the dispatches of a submission that share a writable buffer.
Mantle.emitloophead!(::WebGPURecorder, ::Mantle.Plan, ::AbstractUnitRange) = nothing

"""
Hand the piece to the page. The last piece ends with the readbacks' mirror copies.
The packed arguments go first, ahead of the first replay; a re-recording's are
written behind the replays already submitted, which WebGPU completes first.
"""
function Mantle.closerecording!(e::WebGPURecorder, pl::Mantle.Plan)
    d = pl.graph.dev
    last(e.passes) == length(pl.passes) && append!(e.commands, mirrorcopies(d, pl))
    am = pl.args
    if am !== nothing
        a = am.store::WebGPUArgs
        post(d, "writeBuffer", d.id, a.id, 0, a.host)
    end
    id = lock(() -> (d.nbuffers += 1), d.lock)
    post(d, "record", d.id, id, e.commands)
    return WebGPURecording(d, id)
end

Mantle.release!(r::WebGPURecording) = (post(r.dev, "forget", r.dev.id, r.id); nothing)

const DispatchSample = @NamedTuple{name::String, pipeline::Int, ns::Union{Nothing,Float64}}

"""Run `f` and return its result and GPU timestamps for recorded dispatches it submits on `d`.

Each measured dispatch runs in its own compute pass. The replay waits for query
readback, so use ordinary replays to measure end-to-end throughput.
"""
function profiledispatches(f, d::WebGPUDevice)
    "timestamp-query" in d.info["features"] || throw(ArgumentError("WebGPU timestamp queries are unavailable"))
    samples = DispatchSample[]
    result = task_local_storage(f, profiledispatches, (d, samples))
    return (; result, samples)
end

"""
One submission of recorded pieces, in order: a loop body's pieces once per
iteration (core's `submissionbatches`), all encoded into one command buffer, behind
the open batch, which holds what was launched before the run and the run's host
stores.
"""
function Mantle.submitrecording!(d::WebGPUDevice, pieces::Union{Tuple{WebGPURecording}, Vector{WebGPURecording}}, _)
    commands, args = takebatch!(d)
    recids = [r.id for r in pieces]
    capture = get(task_local_storage(), profiledispatches, nothing)
    if capture !== nothing && capture[1] === d
        rows = call(d, "profileReplay", d.id, recids, commands, d.args, args)
        for row in rows
            push!(capture[2], (name = row["name"], pipeline = row["pipeline"], ns = row["ns"]))
        end
    else
        post(d, "replay", d.id, recids, commands, d.args, args)
    end
    submitted!(d)
    return nothing
end

"""
A run's end, one submission: the open batch (launches made before the run, then
the run's host stores), then the recording, or for a walked plan the batch the
walk filled and the readbacks' mirror copies.
"""
function submitrun!(d::WebGPUDevice, pl::Mantle.Plan)
    if pl.recording === nothing
        foreach(cmd -> enqueue!(d, cmd), mirrorcopies(d, pl))
        Mantle.flush!(d)
    else
        Mantle.submitplan!(d, pl, nothing, WebGPURecording)
    end
    return Mantle.fence(d)
end
Mantle.closerun!(d::WebGPUDevice, pl::Mantle.Plan, ::Union{WebGPURun, Nothing}) = submitrun!(d, pl)

end
