"""
Mantle's ROCm backend on AMDGPU.jl. Verbs only: every decision (schedule,
liveness, placement, barriers, the pool, `record!`, `run!`) is core's, and this
module answers the backend vocabulary declared in `graph/backend.jl`.
[`deviceview`](@ref) returns a borrowed `ROCArray`, so rocBLAS, MIOpen and any
`ROCArray` code read a Mantle region without copying. A recording is a HIP graph
captured from the device stream, and a replay allocates no host memory.

An extension because AMDGPU.jl is optional and loads beside the Vulkan backend.
Every method is written `Mantle.x`: a missed qualifier silently defines a new
function in this module instead of adding a method.
"""
module MantleROCmExt

import AMDGPU
import Adapt
import GPUArrays
import KernelAbstractions as KA
import KernelInterface as KI
import Mantle
import LinearAlgebra
import Libdl
using Mantle: Pool, Buffers, Images, Persistent, TransientBuffer, DeviceArray,
    DeviceInfo, ROCmAPI, register_backend!, selectdevice, adopt!
using KernelInterface: DeviceCaps, MatrixShape

const Managed = AMDGPU.Managed
const HIPBuffer = AMDGPU.Mem.HIPBuffer

# ── device ────────────────────────────────────────────────────────────────────

"""
    Device(ROCmAPI())
    Device(AMDGPU.ROCBackend())

A HIP device: one `HIPDevice`, one owned stream, one pool, one timeline. Every
verb that touches the GPU makes the device stream the task's current stream and
does not restore the previous one. AMDGPU.jl streams are task-local; work on a
second stream breaks the single in-order stream `needs_transition(::ROCmAPI, …)`
relies on (aliased transients corrupt silently, `waitidle` waits on the wrong
stream), and an argument last used on another stream makes AMDGPU.jl
synchronise, which invalidates a capture. The timeline counts retirements, not
submissions: AMDGPU.jl submits a `Launch`, so there is no event to signal.
"""
mutable struct ROCmDevice <: Mantle.Device
    dev::AMDGPU.HIPDevice
    stream::AMDGPU.HIP.HIPStream
    # hipBLASLt handle of this device, passed into calls, never looked up globally.
    blaslt::Ptr{Cvoid}
    pool::Pool
    # `next` is the fence a retirement gets; `completed` is the highest value a
    # real synchronise has covered.
    next::UInt64
    completed::UInt64
    # `hipStreamSynchronize` when true, AMDGPU.jl's polling wait when false; see
    # `waitidle`.
    blocking::Bool
    caps::Union{Nothing,DeviceCaps}
    accesses::Mantle.AccessCache
end

"""
The hipBLASLt next to the HIP runtime AMDGPU.jl uses, or `""`. Not a loader
search frozen at precompile time, which can pick another ROCm installation (a
PyTorch wheel's) and load a second HIP runtime into the process.
"""
function hipblaslt_library()
    isempty(AMDGPU.libhip) && return ""
    dir = dirname(AMDGPU.libhip)
    for name in ("libhipblaslt.so.1", "libhipblaslt.so.0", "libhipblaslt.so")
        path = joinpath(dir, name)
        isfile(path) && return path
    end
    return ""
end

"""The `dlopen` handle of [`hipblaslt_library`](@ref), set in `__init__`; `C_NULL` without one."""
const HIPBLASLT = Ref{Ptr{Cvoid}}(C_NULL)

"""A hipBLASLt entry point, by name."""
blaslt(sym::Symbol) = Libdl.dlsym(HIPBLASLT[], sym)

function hipblaslt_handle()
    HIPBLASLT[] == C_NULL && return C_NULL
    h = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall(blaslt(:hipblasLtCreate), Cint, (Ref{Ptr{Cvoid}},), h)
    return status == 0 ? h[] : C_NULL
end

function ROCmDevice(dev::AMDGPU.HIPDevice = AMDGPU.device())
    # `HIPStream()` creates on the current device, so create on `dev` explicitly.
    stream = dev == AMDGPU.device() ? AMDGPU.HIPStream() :
             AMDGPU.device!(AMDGPU.HIPStream, dev)
    d = ROCmDevice(dev, stream, C_NULL, Pool(), UInt64(0), UInt64(0), true, nothing,
                   Mantle.AccessCache())
    adopt!(d)
    d.blaslt = hipblaslt_handle()
    return d
end

"""Make this device's stream the current task's. Idempotent; does not restore the
previous stream (see `ROCmDevice`)."""
@inline function Mantle.adopt!(d::ROCmDevice)
    AMDGPU.stream() == d.stream || AMDGPU.stream!(d.stream)
    return d
end

# Core keeps the default device (`runtime/lifecycle.jl`). ROCm has no validation
# layer, so the only configuration is the default one.
function Mantle.makedevice(::ROCmAPI, select, debug::Mantle.DebugConfig)
    debug == Mantle.DebugConfig() || throw(ArgumentError(
        "ROCm: there is no validation layer to configure; leave `debug` at its default"))
    return ROCmDevice(select === nothing ? AMDGPU.device() :
                      AMDGPU.devices()[selectdevice(select, Mantle.devices(ROCmAPI()))])
end

# `integrated` drives the selector's ranking; on an APU device memory is system
# memory.
Mantle.devices(::ROCmAPI) =
    [DeviceInfo(i, AMDGPU.HIP.name(d),
                AMDGPU.HIP.properties(d).integrated == 1 ? :integrated : :discrete,
                "ROCm $(AMDGPU.HIP.runtime_version())")
     for (i, d) in enumerate(AMDGPU.devices())]

# The Mantle device of AMDGPU.jl's backend, which is what `KA.get_backend` of a
# `ROCArray` returns.
Mantle.Device(::AMDGPU.ROCBackend) = Mantle.Device(ROCmAPI())
# A backend's `==` is "the same device" (see the Metal backend); `isequal` stays
# identity, so hashing asks nothing of AMDGPU.jl's type.
Base.:(==)(b::AMDGPU.ROCBackend, d::ROCmDevice) = Mantle.Device(b) === d
Base.:(==)(d::ROCmDevice, b::AMDGPU.ROCBackend) = b == d
Base.isequal(::AMDGPU.ROCBackend, ::ROCmDevice) = false
Base.isequal(::ROCmDevice, ::AMDGPU.ROCBackend) = false
Mantle.pool(d::ROCmDevice) = d.pool
Mantle.accesscache(d::ROCmDevice) = d.accesses
Mantle.devicebuffertype(::ROCmDevice, ::Type{T}, N::Int) where {T} =
    AMDGPU.Device.ROCDeviceArray{T,N,1}
Mantle.argtype(d::ROCmDevice, y) = Core.Typeof(KI.argconvert(d, y))
Mantle.isdevicearray(::AMDGPU.Device.ROCDeviceArray) = true

function Mantle.kernelinterpreter(d::ROCmDevice, f, tt)
    config = AMDGPU.Compiler.compiler_config(d.dev)
    source = Mantle.GPUCompiler.methodinstance(typeof(f), tt)
    return Mantle.GPUCompiler.get_interpreter(Mantle.GPUCompiler.CompilerJob(source, config))
end
Mantle.devicename(d::ROCmDevice) = AMDGPU.HIP.name(d.dev)
Mantle.syncbackend(::ROCmDevice) = ROCmAPI()

"""
    waitidle(dev)

Wait for everything submitted to the device stream. Uses `hipStreamSynchronize`
when `dev.blocking` (the default); AMDGPU.jl's polling wait costs about 1.3 ms
per call with work outstanding. Set `dev.blocking = false` for kernels that make
hostcalls (device printf, device allocation, a throw), which a blocking wait can
deadlock.
"""
Mantle.waitidle(d::ROCmDevice) =
    (AMDGPU.synchronize(d.stream; blocking = d.blocking); nothing)

Mantle.awaitwrites(d::ROCmDevice) = Mantle.waitidle(d)

# ── allocation ───────────────────────────────────────────────────────────────
#
# `hipMalloc` memory has no usage flags or storage modes that matter here, so a
# constraint is `nothing` and every block hosts every request. An arena the CPU
# reads goes through `download`, as on a discrete card.

Mantle.constraintof(::ROCmDevice, ::Union{Buffers,Images,Persistent}, ts) = nothing
Mantle.compatible(::ROCmDevice, blk, req) = true
Mantle.mergeconstraints(::ROCmDevice, kind, a, b) = nothing

"""
    rawalloc(dev, kind, bytes, constraint) -> Managed{HIPBuffer}

One device allocation for a pool block. A `Managed`, because it records the
stream the buffer was last used on and `convert(Ptr, ::Managed)` synchronises
when that is not the current stream; a bare `HIPBuffer` would allow reading
memory a copy on another stream has not finished writing. `max(bytes, 1)`: a
zero-length `hipMalloc` returns `C_NULL`.
"""
function Mantle.rawalloc(d::ROCmDevice, ::Union{Buffers,Images,Persistent},
                         bytes::Int, constraint)
    adopt!(d)
    buf = HIPBuffer(max(bytes, 1); stream = d.stream)
    buf.ptr == C_NULL && throw(OutOfMemoryError())
    return Managed(buf; stream = d.stream)
end

"""
Release a block's device memory now, through AMDGPU.jl's `pool_free`: it falls
back to the default stream if the buffer's stream is gone, and accounts the
bytes back to `memory_stats`, so `AMDGPU.info()` agrees with `Mantle.capacity`.
"""
Mantle.rawfree(d::ROCmDevice, mg::Managed) = (adopt!(d); AMDGPU.pool_free(mg); nothing)


"""Free device memory now (`hipMemGetInfo`), the bound the placer checks an arena against."""
Mantle.capacity(::ROCmDevice) = AMDGPU.free()
Mantle.maxalloc(d::ROCmDevice) = Int(AMDGPU.HIP.properties(d.dev).totalGlobalMem)

# ── the timeline ─────────────────────────────────────────────────────────────

function Mantle.fence(d::ROCmDevice)
    d.next += UInt64(1)
    return d.next
end

Mantle.passed(d::ROCmDevice, f) = UInt64(f) <= d.completed

# A token no retirement has been handed: nothing will ever cover it, and core's
# `waitfor!` makes the refusal an error.
Mantle.issued(d::ROCmDevice) = d.next

"""
Wait until the timeline reaches `f`, with a full stream synchronise: this backend
cannot observe the submission that reads a region. `next` is read before the
wait, so the sync is credited only with the retirements it covered.
"""
function Mantle.awaittoken!(d::ROCmDevice, ::UInt64)
    upto = d.next
    Mantle.waitidle(d)
    d.completed = max(d.completed, upto)
    return nothing
end

# ── regions as arrays ────────────────────────────────────────────────────────

"""
    rocview(mg, T, dims, byteoffset) -> ROCArray{T}

Borrow a bounded, aligned span of a pool block as a `ROCArray` whose `DataRef`
never frees it; keep the owning region alive. Checks the resulting pointer, so a
disagreement about AMDGPU's offset unit fails at once.
"""
function rocview(mg::Managed, ::Type{T}, dims::Dims{N}, byteoffset::Int) where {T,N}
    Mantle.checkspan(Int(sizeof(mg)),byteoffset,Mantle.checkedbytes(T,dims))
    # Element alignment, rather than element size, constrains the offset.
    byteoffset % Base.datatype_alignment(T) == 0 || throw(ArgumentError(
        "offset $byteoffset is not a multiple of $T's alignment, " *
        "$(Base.datatype_alignment(T))"))
    ref = GPUArrays.DataRef(_ -> nothing, mg)
    a = AMDGPU.ROCArray{T,N}(ref, dims; offset = byteoffset)
    want = convert(Ptr{T}, mg.mem) + byteoffset
    pointer(a) == want || throw(ErrorException(
        "AMDGPU's ROCArray placed a view at $(pointer(a)) where Mantle's " *
        "region is at $want. `ROCArray.offset` is being read in a unit this " *
        "backend does not agree with — see `rocview`."))
    return a
end

"""
    deviceview(dev, a::DeviceArray) -> ROCArray

A borrowed `ROCArray` over `a`'s region, so rocBLAS, MIOpen, rocFFT and any
`ROCArray` code use it without copying.
"""
Mantle.deviceview(::ROCmDevice, a::DeviceArray{T,N}) where {T,N} =
    rocview(Mantle.memoryof(a)::Managed, T, size(a), Mantle.offset(a))

"""
A placed transient's slice of the arena, as the borrowed `ROCArray` a launch
passes. Built once; core's `storage(::TransientBuffer)` returns it.
"""
Mantle.deviceslice(::ROCmDevice, ::Type{T}, dims::Dims, slab::Managed,
                   offset::Int) where {T} = rocview(slab, T, dims, offset)

"""
256 bytes. HIP publishes no storage offset alignment; this keeps aliased
transients out of each other's 128-byte cache lines and satisfies every element
type's natural alignment, so `rocview`'s arithmetic is exact.
"""
Mantle.alignment(::ROCmDevice, ::TransientBuffer) = 256

# ── moving bytes ─────────────────────────────────────────────────────────────
#
# `hostspan` is not implemented: `hipMalloc` memory is not mapped into the
# process even on an APU, so `hostview` would segfault on first touch. The
# transfer verbs use `copyto!`, which is `hipMemcpyAsync` on the device stream.

function Mantle.upload!(d::ROCmDevice, a::DeviceArray, first::Integer,
                        data::AbstractVector)
    adopt!(d)
    copyto!(Mantle.deviceview(d, a), Int(first), data, 1, length(data))
    return a
end

"""
    download(dev, a) -> Vector

`a`'s contents on the host. Synchronises first, as core's `hostdownload` does,
or a copy queued behind a running kernel returns the old contents.
"""
function Mantle.download(d::ROCmDevice, a::DeviceArray)
    adopt!(d)
    Mantle.awaitwrites(d)
    return Array(Mantle.deviceview(d, a))
end

function Mantle.devicecopy!(d::ROCmDevice, dst::DeviceArray{T}, src::DeviceArray{T},
                            n::Integer) where {T}
    adopt!(d)
    copyto!(Mantle.deviceview(d, dst), 1, Mantle.deviceview(d, src), 1, Int(n))
    return dst
end

# ── what a dispatch compiles to ──────────────────────────────────────────────

"""
One compiled dispatch: the HIP kernel, its context and arguments, and the launch
geometry. Everything KernelAbstractions would derive per launch is derived once
at `Pipelines` time, which also makes a HIP graph capturable: compiling inside
`hipStreamBeginCapture` invalidates the capture. Callable, so core's
`emitdispatch!(::Immediate, d, name)` runs it.
"""
struct ROCmCompiledDispatch{K,C,A<:Tuple,R<:Tuple,G,N}
    kernel::K
    ctx::C
    args::A
    # The graph resources `args` were resolved from; see `checkresolved`.
    raw::R
    # `Int` or a tuple of up to three. A `@kernel` reads its index from the
    # `CompilerMetadata` in `ctx`, so its launch is flat. A macro-free kernel
    # reads `get_group_id()` directly, so its launch keeps every axis;
    # flattening silently made multi-dimensional dispatches one-dimensional.
    groupsize::G
    gridsize::N
end

function (d::ROCmCompiledDispatch)()
    # A zero grid launches nothing (an emptied queue round is ordinary), as in
    # core's `Launch`.
    prod(d.gridsize) == 0 && return nothing
    # `ctx === nothing` is a KernelInterface kernel, a plain function without
    # `CompilerMetadata`. The field type is concrete, so this specialises away.
    if d.ctx === nothing
        d.kernel(d.args...; groupsize = d.groupsize, gridsize = d.gridsize)
    else
        d.kernel(d.ctx, d.args...; groupsize = d.groupsize, gridsize = d.gridsize)
    end
    return nothing
end

# No argument memory and no indirect slot: arguments are passed directly, and a
# `DeviceRange` launches its ceiling.
Mantle.argsize(::ROCmCompiledDispatch) = 0
Mantle.indirectindex(::ROCmCompiledDispatch) = 0
Mantle.workgroupsize(d::ROCmCompiledDispatch) = UInt32(prod(d.groupsize))

# ── calls ────────────────────────────────────────────────────────────────────
#
# A call is core's `Mantle.Call`, baked by core. It records here because a
# recording is a stream capture and rocBLAS submits to the current stream. A
# body that synchronises or reads back invalidates the capture, which
# `closerecording!` reports. `openrecording` refuses a `Mantle.Launch` (a
# device-only count read on the host), not a `Call`.

"""
    kicompile(compile, dev, dispatch) -> ROCmCompiledDispatch

Compile a macro-free kernel through KernelInterface (`KI.kernel_function` is
`hipfunction` here) and resolve its launch geometry with
`Mantle.ki_launch_extents`, once, because a capture cannot contain host work.
The occupancy limit must be shaped to the ndrange; reading the size
`KA.launch_config` returns instead skipped the autotune and cost 29% of SAM
2.1's encoder.
"""
function kicompile(c::Mantle.Compile{ROCmDevice}, dev::ROCmDevice, d::Mantle.Dispatch)
    args = Mantle.resolve(dev, d.args)
    nd = Mantle.dispatchrange(d.ndrange)
    kern = Mantle.kikernel(d.kernel, dev, args)
    wg, ng = Mantle.ki_launch_extents(kern, Mantle.groupdims(nd), d.group)
    # The tuples, not their products (see `ROCmCompiledDispatch`): the geometry
    # a `KI.Kernel` call resolves before `KI.launch`. `kern.kern.kernel` is the
    # `HIPKernel`; a plain function captures nothing, so it launches alone.
    return ROCmCompiledDispatch(kern.kern.kernel, nothing, args, d.args, wg, ng)
end

"""
    compile_dispatch(compile, dispatch, argoff, indirect)

Compile one dispatch into something the walk runs and a capture records. A
dispatch over a `DeviceRange` without `max` goes back to core's `bake` as a
`Launch`: its count is read on the host, which a capture forbids, so
`openrecording` refuses plans holding one.
"""
function Mantle.compile_dispatch(c::Mantle.Compile{ROCmDevice}, d::Mantle.Dispatch,
                                 argoff::Int, indirect::Int)
    dev = c.graph.dev
    d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing &&
        return Mantle.bake(c, d)
    # A call (no ndrange): core's `bake` returns a `Mantle.Call`, capturable here.
    Mantle.iscall(d) && return Mantle.bake(c, d)
    # A plain-function kernel goes through KernelInterface.
    Mantle.buildskernel(d.kernel, dev) || return kicompile(c, dev, d)
    obj = Mantle.kernelfor(d, dev)
    args = Mantle.resolve(dev, d.args)
    # `callgroup`, not `nothing`: a `DynamicSize` kernel launched with
    # `workgroupsize = wg` carries it nowhere else, and dropping it lets the
    # autotune pick a size a body reading its size from a `Val` disagrees with.
    # A `DynamicSize` body launched without a requested `group` is repartitioned
    # by its occupancy, as `ROCKernels` does per launch. `group` is the caller's
    # request, not `KA.launch_config`'s result, which substitutes a default and
    # would skip the autotune.
    kernel, shape = Mantle.tunedlaunch(ctx -> AMDGPU.hipfunction(obj.f, hiptypes(ctx, args)),
                                       (k, nd) -> KI.threads_to_workgroupsize(
                                           AMDGPU.launch_configuration(k).groupsize, nd),
                                       obj, Mantle.dispatchrange(d.ndrange),
                                       Mantle.callgroup(obj, d.group))
    return ROCmCompiledDispatch(kernel, shape.ctx, args, d.args,
                                Mantle.groupthreads(shape), Mantle.groupcount(shape))
end

hiptypes(ctx, args) =
    Base.to_tuple_type(map(Core.Typeof, map(AMDGPU.rocconvert, (ctx, args...))))

# ── a recording, and a run ───────────────────────────────────────────────────
#
# A recording is a `hipGraph`: the stream is in capture mode while the plan's
# ordinary walk, through core's `Immediate` emitter, submits into it.

"""
`true`: every plan this backend accepts is recordable, and `openrecording`
records or throws (profiled plans are excluded by `recordable`).
`DNNKernels.recordplan` refuses to record when this is `false`.
"""
Mantle.recordsplans(::ROCmDevice) = true

# ── profiling ────────────────────────────────────────────────────────────────

# A profiled plan is walked every run instead of replayed: a captured graph has no
# pass walk to bracket with timing events.
Mantle.recordsprofiled(::ROCmDevice) = false

# Timing HIP events around each pass on the device stream; waiting on the stop
# event serialises a profiled run.
function Mantle.passbegin!(pl::Mantle.Plan{ROCmDevice}, ::Mantle.Immediate, ::Integer)
    adopt!(pl.graph.dev)
    return AMDGPU.HIP.HIPEvent(pl.graph.dev.stream; timing = true)
end
function Mantle.passend!(pl::Mantle.Plan{ROCmDevice}, ::Mantle.Immediate, ::Integer, start)
    stop = AMDGPU.HIP.HIPEvent(pl.graph.dev.stream; timing = true)
    AMDGPU.HIP.hipEventSynchronize(stop)
    return AMDGPU.HIP.elapsed(start, stop) * 1e9
end

"""
`true`: a call runs and records here, because a recording is a stream capture
and the library's kernels land on the capturing stream. A body that
synchronises or reads back invalidates the capture; `closerecording!` reports it.
"""
Mantle.runscalls(::ROCmDevice) = true

# ── fused library GEMM ──────────────────────────────────────────────────────

"""A hipBLASLt matmul with the bias epilogue, carrying its owning device handle."""
struct HipBLASLtBias
    handle::Ptr{Cvoid}
end

Mantle.argument_usage(::Type{HipBLASLtBias},
                      ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

@inline function ltcheck(status::Integer, where::AbstractString)
    status == 0 || throw(ErrorException("hipBLASLt $where failed with status $status"))
    return nothing
end

function (call::HipBLASLtBias)(D, A, B, bias)
    M, K = size(A)
    Kb, N = size(B)
    K == Kb && size(D) == (M, N) && length(bias) == M ||
        throw(DimensionMismatch("hipBLASLt bias GEMM: $(size(A)) * $(size(B)), " *
                                "bias $(size(bias)), destination $(size(D))"))

    desc = Ref{Ptr{Cvoid}}(C_NULL)
    ad = Ref{Ptr{Cvoid}}(C_NULL)
    bd = Ref{Ptr{Cvoid}}(C_NULL)
    dd = Ref{Ptr{Cvoid}}(C_NULL)
    try
        # HIPBLAS_COMPUTE_32F = 2, HIP_R_32F = 0, HIP_R_16F = 2.
        ltcheck(ccall(blaslt(:hipblasLtMatmulDescCreate), Cint,
                      (Ref{Ptr{Cvoid}}, Cint, Cint), desc, 2, 0), "descriptor creation")
        epilogue = Ref{UInt32}(4) # HIPBLASLT_EPILOGUE_BIAS
        ltcheck(ccall(blaslt(:hipblasLtMatmulDescSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 2, epilogue, sizeof(UInt32)), "bias epilogue")
        biasptr = Ref{Ptr{Cvoid}}(Ptr{Cvoid}(pointer(bias)))
        ltcheck(ccall(blaslt(:hipblasLtMatmulDescSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 3, biasptr, sizeof(Ptr{Cvoid})), "bias pointer")
        biastype = Ref{Int32}(2)
        ltcheck(ccall(blaslt(:hipblasLtMatmulDescSetAttribute), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 4, biastype, sizeof(Int32)), "bias type")
        for (layout, rows, cols, ld) in
                ((ad, M, K, M), (bd, K, N, K), (dd, M, N, M))
            ltcheck(ccall(blaslt(:hipblasLtMatrixLayoutCreate), Cint,
                          (Ref{Ptr{Cvoid}}, Cint, UInt64, UInt64, Int64),
                          layout, 2, rows, cols, ld), "matrix layout")
        end

        alpha = Ref{Float32}(1)
        beta = Ref{Float32}(0)
        GC.@preserve D A B bias begin
            ltcheck(ccall(blaslt(:hipblasLtMatmul), Cint,
                          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                           Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                           Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t,
                           Ptr{Cvoid}),
                          call.handle, desc[], alpha, pointer(A), ad[], pointer(B), bd[],
                          beta, pointer(D), dd[], pointer(D), dd[], C_NULL, C_NULL, 0,
                          AMDGPU.stream().stream), "matmul")
        end
    finally
        for layout in (ad[], bd[], dd[])
            layout == C_NULL || ccall(blaslt(:hipblasLtMatrixLayoutDestroy),
                                      Cint, (Ptr{Cvoid},), layout)
        end
        desc[] == C_NULL || ccall(blaslt(:hipblasLtMatmulDescDestroy),
                                  Cint, (Ptr{Cvoid},), desc[])
    end
    return D
end

# ── MIOpen convolution ──────────────────────────────────────────────────────

"""
MIOpen forward convolution as a pass member. Descriptors are built at
declaration, the workspace is a graph transient, and the algorithm search runs
on the first execution, the warm-up `openrecording` gives every call before
capture. AMDGPU.jl's `MIOpen.convolution!` allocates a workspace per call, which
a capture cannot replay.
"""
mutable struct MIOpenConv
    xdesc::AMDGPU.MIOpen.TensorDescriptor
    wdesc::AMDGPU.MIOpen.TensorDescriptor
    ydesc::AMDGPU.MIOpen.TensorDescriptor
    cdesc::AMDGPU.MIOpen.ConvolutionDescriptor
    wsize::Int
    algo::Union{Nothing,AMDGPU.MIOpen.miopenConvFwdAlgorithm_t}
end

Mantle.argument_usage(::Type{MIOpenConv}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.WRITE)

function (c::MIOpenConv)(y, x, w, workspace)
    (; handle) = AMDGPU.MIOpen.lib_state()
    if c.algo === nothing
        perf = AMDGPU.MIOpen.find_conv_algo(AMDGPU.MIOpen.miopenConvFwdAlgorithm_t;
            handle, workspace, a = x, a_desc = c.xdesc, b = w, b_desc = c.wdesc,
            conv_desc = c.cdesc, c = y, c_desc = c.ydesc)
        c.algo = perf.fwd_algo
    end
    AMDGPU.MIOpen.miopenConvolutionForward(handle, Ref{Float32}(1f0), c.xdesc.handle, x,
        c.wdesc.handle, w, c.cdesc.handle, c.algo, Ref{Float32}(0f0), c.ydesc.handle, y,
        workspace, c.wsize)
    return nothing
end

"""`out .= y`, converting: the wider result a convolution on narrower operands owes."""
struct Widen end
Mantle.argument_usage(::Type{Widen}, ::Type{<:Tuple{Any,Any}}) = (Mantle.WRITE, Mantle.READ)
(::Widen)(out, y) = (out .= y; nothing)

"""A dense MIOpen descriptor for an array of size `dims`, in the reversed (WHCN) layout."""
function miodesc(::Type{T}, dims::Dims{N}) where {T,N}
    strides = cumprod((1, dims[1:end-1]...))
    AMDGPU.MIOpen.TensorDescriptor(AMDGPU.MIOpen.miopen_data_type(T), N,
                                   Int32[reverse(dims)...], Int32[reverse(strides)...])
end

"""
2-D convolution through [`MIOpenConv`](@ref) for Float16 or Float32 operands of
one type; returns `nothing` when it does not apply. A wider result is convolved
into an operand-typed transient and widened, so it has the operands' precision.
Bias and activation stay with the caller's passes. Faster than DNNKernels'
im2col and GEMM, which took about 1.2 s of FLUX.2-klein's 1.6 s VAE decode.
"""
function Mantle.native_conv2d_dispatch!(d::ROCmDevice, g, out, x, w; bias = nothing,
                                        stride, pad, dilation, groups,
                                        epilogue = identity, transposed = false, name)
    transposed && return nothing
    T = eltype(x)
    (T === Float16 || T === Float32) && eltype(w) === T || return nothing
    (eltype(out) === Float16 || eltype(out) === Float32) || return nothing
    Mantle.denseoperands(out, x, w) || return nothing
    ndims(out) == ndims(x) == ndims(w) == 4 || return nothing
    y = eltype(out) === T ? out : Mantle.Transient.Buffer(g, T, size(out))
    xd, wd, yd = miodesc(T, size(x)), miodesc(T, size(w)), miodesc(T, size(out))
    cdesc = AMDGPU.MIOpen.ConvolutionDescriptor(2; padding = pad, stride, dilation, groups)
    adopt!(d)
    wsize = Int(AMDGPU.MIOpen.get_workspace_size(AMDGPU.MIOpen.miopenConvFwdAlgorithm_t;
        handle = AMDGPU.MIOpen.lib_state().handle, a_desc = xd, b_desc = wd,
        conv_desc = cdesc, c_desc = yd))
    workspace = Mantle.Transient.Buffer(g, UInt8, (max(wsize, 256),))
    Mantle.dispatch!(g, MIOpenConv(xd, wd, yd, cdesc, wsize, nothing), (y, x, w, workspace); name)
    y === out || Mantle.dispatch!(g, Widen(), (out, y); name = "$(name).widen")
    return (; bias = false, epilogue = false)
end

# ── strided-batched GEMM ─────────────────────────────────────────────────────

"""
rocBLAS strided-batched GEMM as a pass member: each plane of `C` (first two axes
the matrix, the rest the batch) is `α op(A) op(B)`, and with `coldiv` each result
column is then divided by its value, the normalisation an unnormalised softmax
leaves. Used instead of DNNKernels' scalar `attn_scores_b*` fallback.
"""
struct RocBLASBatched{T}
    ta::Char
    tb::Char
    alpha::T
end

Mantle.argument_usage(::Type{<:RocBLASBatched}, ::Type{<:Tuple{Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ)
Mantle.argument_usage(::Type{<:RocBLASBatched}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

"The matrix planes of `x`: its first two axes, and every trailing one as the batch."
planes(x) = reshape(x, size(x, 1), size(x, 2), :)

function (c::RocBLASBatched{T})(C, A, B) where {T}
    AMDGPU.rocBLAS.gemm_strided_batched!(c.ta, c.tb, c.alpha, planes(A), planes(B), zero(T), planes(C))
    return nothing
end

function (c::RocBLASBatched)(C, A, B, coldiv)
    c(C, A, B)
    P = planes(C)
    P ./= reshape(coldiv, 1, size(P, 2), size(P, 3))
    return nothing
end

"""
The batched product through [`RocBLASBatched`](@ref) for Float32 (fp16 attention
uses DNNKernels' fused kernels); returns whether it dispatched. Declines a
transposed `B`: rocBLAS' NT kernel took 492 ms where DNNKernels'
`attn_apply_b32!` takes 62 ms on FLUX.2-klein's VAE.
"""
function Mantle.native_batched_gemm_dispatch!(d::ROCmDevice, g, out, A, B;
                                               transpose_a = false, transpose_b = false,
                                               alpha = 1, coldiv = nothing, name)
    transpose_b && return false
    T = eltype(out)
    T === Float32 && eltype(A) === T && eltype(B) === T || return false
    coldiv === nothing || eltype(coldiv) === T || return false
    Mantle.denseoperands(out, A, B, coldiv) || return false
    batch(x) = prod(size(x)[3:end]; init = 1)
    batch(out) == batch(A) == batch(B) || return false
    call = RocBLASBatched(transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N', T(alpha))
    Mantle.dispatch!(g, call, coldiv === nothing ? (out, A, B) : (out, A, B, coldiv); name)
    return true
end

function Mantle.librarygemm(d::ROCmDevice, out, A, B, bias, epilogue)
    d.blaslt == C_NULL && return nothing
    epilogue === identity || return nothing
    bias === nothing && return nothing
    ndims(out) == ndims(A) == ndims(B) == 2 || return nothing
    eltype(out) === eltype(A) === eltype(B) === eltype(bias) === Float16 ||
        return nothing
    size(A, 2) == size(B, 1) || return nothing
    size(out) == (size(A, 1), size(B, 2)) || return nothing
    ndims(bias) == 1 && length(bias) == size(out, 1) || return nothing
    return HipBLASLtBias(d.blaslt)
end

"""
    openrecording(dev, plan, passes) -> Immediate

Put the device stream into capture mode so the walk of `passes` is recorded into
a HIP graph; a partitioned `record!` calls this once per piece. Returns `nothing`
for a profiled plan. Every call in the piece runs once first, because rocBLAS
creating its handle inside a capture segfaults. Throws for a `DeviceRange`
dispatch without `max`, whose host-side count read would invalidate the capture
(that needs `emitkernel!`, not implemented here).
"""
function Mantle.openrecording(d::ROCmDevice, pl::Mantle.Plan, passes::AbstractUnitRange)
    # A profiled plan is walked every run; see `Mantle.recordable` above.
    pl.profile && return nothing
    checkresolved(d, pl)
    piece = view(pl.passes, passes)
    any(pp -> any(x -> x isa Mantle.Launch, pp.dispatches), piece) && throw(ArgumentError(
        "record!: this plan dispatches over a `DeviceRange` with no ceiling, whose " *
        "count is read on the host. A capture is invalidated by a host read, so " *
        "there is no graph to record — give the range a `max`, or run the plan " *
        "without recording it."))
    adopt!(d)
    # Run every call of this piece once before capture: rocBLAS creates its
    # handle lazily, and creating it on a capturing stream segfaults in
    # libamdhip64 instead of invalidating the capture. The writes land in
    # plan-owned bytes the replay rewrites; a gated pass's call runs too, and
    # its gate is still evaluated per run inside the graph.
    for pp in piece, cd in pp.dispatches
        cd isa Mantle.Call && cd()
    end
    # Take AMDGPU stream ownership of every argument before capture: an array
    # last used on another stream (constant folding across short plans) would
    # make the launch synchronise its old owner, which a capture rejects with
    # `hipErrorStreamCaptureUnsupported`.
    for pp in piece, cd in pp.dispatches
        cd isa ROCmCompiledDispatch || continue
        # The one-argument `rocconvert` passes `stream = nothing` and takes no
        # ownership, so pass the launch stream.
        map(arg -> AMDGPU.rocconvert(arg, d.stream), cd.args)
    end
    # Drain the stream, warm-up included: work in flight would otherwise become
    # a dependency the graph waits for once and never again.
    Mantle.waitidle(d)
    AMDGPU.HIP.begincapture!(d.stream)
    return Mantle.Immediate()
end

"""
    closerecording!(emitter, plan) -> ROCmRecording

Close the capture and instantiate it. Throws if the capture was invalidated or
holds allocation or free nodes.
"""
function Mantle.closerecording!(::Mantle.Immediate, pl::Mantle.Plan{ROCmDevice})
    d = pl.graph.dev
    graph = endcapture!(d)
    graph === nothing && throw(ErrorException(
        "record!: the capture was invalidated, so there is no graph. Something " *
        "in the walk reached the host — a synchronise, an allocation on the " *
        "capturing stream, or a kernel that still had to be compiled."))
    mem = memorynodes(graph)
    mem == 0 || throw(ErrorException(
        "record!: the capture holds $mem allocation or free node(s). A plan owns " *
        "its memory, so its walk allocates and frees nothing: either a call in the " *
        "plan allocates per invocation, or something outside the walk allocated " *
        "or freed on this device's stream while it was capturing. A graph that " *
        "frees memory it did not allocate segfaults in `hipGraphLaunch`, so the " *
        "recording is refused."))
    return ROCmRecording(AMDGPU.HIP.instantiate(graph), graph)
end

# Allocation and free nodes in the graph; a plan's walk must have none. A free of
# memory the graph did not allocate (an array finalised during capture)
# segfaults in `hipGraphLaunch`, so `closerecording!` refuses it at `record!`.
function memorynodes(graph::AMDGPU.HIP.HIPGraph)
    HIP = AMDGPU.HIP
    n = Ref{Csize_t}(0)
    HIP.hipGraphGetNodes(graph, C_NULL, n)
    nodes = Vector{HIP.hipGraphNode_t}(undef, n[])
    HIP.hipGraphGetNodes(graph, nodes, n)
    t = Ref{HIP.hipGraphNodeType}()
    return count(nodes) do node
        HIP.hipGraphNodeGetType(node, t)
        t[] == HIP.hipGraphNodeTypeMemAlloc || t[] == HIP.hipGraphNodeTypeMemFree
    end
end

"""
A plan's recording: the executable graph, and the graph it came from, kept
reachable so its `hipGraphDestroy` finalizer cannot run while the exec lives.
Concrete, so core's `submitplan!` narrows `Plan.recording` to it statically; a
parametric one left every run building a boxed one-piece batch.
"""
struct ROCmRecording
    exec::AMDGPU.HIP.HIPGraphExec
    graph::AMDGPU.HIP.HIPGraph
end

"""
    release!(recording)

A no-op: both HIP handles carry finalizers that destroy them, so destroying here
would free twice. Defined because `invalidate!` calls it under the pool's lock,
where a `MethodError` would surface.
"""
Mantle.release!(::ROCmRecording) = nothing

"""
    abandonrun!(dev, emitter)

Close the capture of a walk that threw. A stream left capturing fails every
later `hipStreamSynchronize` with `hipErrorStreamCaptureUnsupported`.
"""
Mantle.abandonrun!(d::ROCmDevice, ::Mantle.Immediate) = (endcapture!(d); nothing)

"""End the capture if one is open; return the graph, or `nothing` if it was
invalidated. Does not throw, so it is safe while unwinding."""
endcapture!(d::ROCmDevice) =
    AMDGPU.HIP.is_capturing(d.stream) ? AMDGPU.HIP.endcapture!(d.stream) : nothing

"""
    checkresolved(dev, plan)

Refuse to record a plan whose resources moved since it was compiled. A compiled
dispatch holds the resolved `ROCArray`, so after a relocating `resize!` a new
recording would capture the old address; unlike Vulkan there is no argument
memory to patch, and the compiled dispatches are core's structure. The plan must
be built again. One `resolve` per argument per `record!`, nothing per run.
"""
function checkresolved(d::ROCmDevice, pl::Mantle.Plan)
    for pp in pl.passes, disp in pp.dispatches
        disp isa ROCmCompiledDispatch || continue
        for (a, r) in zip(disp.args, disp.raw)
            a isa AMDGPU.ROCArray || continue
            now = Mantle.resolve(d, r)
            now isa AMDGPU.ROCArray && pointer(now) == pointer(a) && continue
            throw(ArgumentError(
                "record!: a resource this plan was compiled against has moved — " *
                "its storage is at $(pointer(now)) where the plan holds " *
                "$(pointer(a)). A compiled dispatch on this backend keeps the " *
                "array rather than a pointer into argument memory, so there is " *
                "nothing to patch and re-recording would capture the old " *
                "address. Build the plan again."))
        end
    end
    return nothing
end

"""
    openrun(dev, plan) -> Immediate

Adopt the device stream for this run: every launch reads `AMDGPU.stream()`, and
`waitidle` and the capture use the device stream.
"""
Mantle.openrun(d::ROCmDevice, pl::Mantle.Plan) = (adopt!(d); Mantle.Immediate())

"""
    closerun!(dev, plan, emitter)

Submit the plan's recording, if any. Two methods, because one with a `Union`
third argument is ambiguous with core's `closerun!(dev, pl, ::Immediate)`. Core
passes `nothing` when the run emitted nothing ahead of its recording.
"""
Mantle.closerun!(d::ROCmDevice, pl::Mantle.Plan, ::Mantle.Immediate) = submitrun!(d, pl)
Mantle.closerun!(d::ROCmDevice, pl::Mantle.Plan, ::Nothing) = submitrun!(d, pl)

function submitrun!(d::ROCmDevice, pl::Mantle.Plan)
    # Through core's `submitplan!`, which iterates a partitioned `RecordingParts`.
    pl.recording === nothing || Mantle.submitplan!(d, pl, nothing, ROCmRecording)
    # The token `waitfor!(plan)` waits on; waiting on it is a full synchronise.
    return Mantle.fence(d)
end

"""
    submitpiece!(dev, recording, emitter)

Launch one captured graph on the device stream. The emitter is always empty
here, since `openrun` returns `Immediate`. A partitioned recording is several of
these in order on the same in-order stream.
"""
function Mantle.submitpiece!(d::ROCmDevice, rec::ROCmRecording, _)
    adopt!(d)
    AMDGPU.HIP.launch(rec.exec, d.stream)
    return nothing
end

"""Close a capture that will not become a graph because the walk threw; a stream
left capturing fails every later synchronise. Core keeps this a separate verb
from [`Mantle.abandonrun!`](@ref); the body here is the same."""
Mantle.abandonrecording!(d::ROCmDevice, ::Mantle.Immediate) = (endcapture!(d); nothing)

"""
    deviceaddress(dev, mg) -> UInt64

The device pointer of the `Managed`'s `HIPBuffer`.
"""
Mantle.deviceaddress(::ROCmDevice, mg::Managed) =
    UInt64(convert(Ptr{UInt8}, mg.mem))

"""
`false`: a captured graph's kernel arguments cannot be rewritten, since
`hipStreamEndCapture` returns no node handles for
`hipGraphExecKernelNodeSetParams`. Core drops the recording on a move instead.
"""
Mantle.patchable(::ROCmDevice) = false

"""
`false`. Core's interpreted `withpredicate` reads the flag on the host
(`gateopen`), a scalar index AMDGPU.jl refuses and a stream
synchronise per gated pass, so a gated pass with fixed-size work is refused at
graph build. A gated pass of only `DeviceRange` dispatches needs `emitkernel!`,
which this backend does not implement, so `repeat!` is unsupported here.
"""
Mantle.supportspredicate(::ROCmDevice) = false

# ── capabilities ─────────────────────────────────────────────────────────────

"""
    caps(dev) -> DeviceCaps

What the portable kernels ask about this device. On gfx11 and gfx12, `coopmat`
reports AMDGPU's native WMMA lowering: one subgroup-scoped 16x16x16 Float16
product with Float32 accumulation, no workgroup-scoped shape. Other
architectures report no matrix shapes.
"""
function Mantle.caps(d::ROCmDevice)
    d.caps === nothing || return d.caps
    p = AMDGPU.HIP.properties(d.dev)
    arch = first(split(AMDGPU.HIP.gcn_arch(d.dev), ':'))
    coopmat = startswith(arch, "gfx11") || startswith(arch, "gfx12")
    return d.caps = DeviceCaps(
        coopmat,
        coopmat ? 16 : 0,
        Int(p.warpSize),                    # subgroup: 32 on gfx11
        Int(p.warpSize),                    # coopmatsubgroup
        Int(p.sharedMemPerBlock),           # sharedbudget
        Int(p.maxThreadsPerBlock),          # workgrouplimit
        # cores: on RDNA HIP counts workgroup processors (two CUs each), half
        # the CU count. Left as reported: CDNA counts CUs, and DepthAnything's
        # tiling, which reads this, runs the same either way.
        Int(p.multiProcessorCount),
        # Resident subgroups per processor. Attention's shared-memory-heavy tile
        # uses it to pick a 16-wave over an 8-wave workgroup (32-35% faster on
        # gfx1151).
        Int(p.maxThreadsPerMultiProcessor) ÷ Int(p.warpSize),
        NTuple{4,Int}[],                    # wggran: no granularity table
        coopmat ? [MatrixShape(Float16, Float32, 16, 16, 16,
                               KI.SubgroupScope())] : MatrixShape[],
    )
end

Mantle.caps(b::AMDGPU.ROCBackend) = Mantle.caps(Mantle.Device(b))

# ── KernelInterface ──────────────────────────────────────────────────────────
#
# A `ROCmDevice` is a KernelInterface backend, as every `Device` is. AMDGPU.jl
# implements the vocabulary for its `ROCBackend`, which works on the task's
# current stream; the device answers each verb as that backend after making its
# own stream current (`adopt!`), so a launch, an allocation or a copy goes to the
# device's stream from any task.

const ROC = AMDGPU.ROCBackend()

KI.allocate(d::ROCmDevice, ::Type{T}, dims::Tuple) where {T} = (adopt!(d); KI.allocate(ROC, T, dims))
KI.synchronize(d::ROCmDevice) = Mantle.waitidle(d)
KI.copyto!(d::ROCmDevice, dest, src) = (adopt!(d); KI.copyto!(ROC, dest, src))
KI.versioninfo(io::IO, ::ROCmDevice) = KI.versioninfo(io, ROC)
KI.supports_shuffle(::ROCmDevice, ::Type{T}) where {T} = KI.supports_shuffle(ROC, T)
KI.argconvert(::ROCmDevice, x) = KI.argconvert(ROC, x)
KI.kernel_function(d::ROCmDevice, f::F, tt::TT = Tuple{}; kwargs...) where {F,TT} =
    (adopt!(d); KI.kernel_function(ROC, f, tt; kwargs...))
KI.record_event(d::ROCmDevice) = (adopt!(d); KI.record_event(ROC))
KI.wait_event(d::ROCmDevice, ev::AMDGPU.HIP.HIPEvent) = (adopt!(d); KI.wait_event(ROC, ev))
for f in (:functional, :supports_float64, :supports_atomics, :supports_unified,
          :supports_subgroups, :shfl_types, :sub_group_reduce_add_types)
    @eval KI.$f(::ROCmDevice) = KI.$f(ROC)
end
# Device queries read the current device, which `adopt!` makes this one.
for f in (:max_work_group_size, :max_work_group_dims, :max_num_groups, :sub_group_size,
          :multiprocessor_count)
    @eval KI.$f(d::ROCmDevice) = (adopt!(d); KI.$f(ROC))
end

# AMDGPU.jl's KernelAbstractions extension tells the compiler a static workgroup
# size; the same kernel built on the device asks it the same way.
KA.compiler_options(obj::KA.Kernel{ROCmDevice,W,N,F}) where {W,N,F} =
    KA.compiler_options(KA.Kernel{AMDGPU.ROCBackend,W,N,F}(ROC, obj.f))

# `Adapt.adapt(dev, x)` uploads to the device.
Adapt.adapt_storage(d::ROCmDevice, x) = (adopt!(d); Adapt.adapt_storage(ROC, x))

# A library call is a pass declared with the three-argument `dispatch!`, which
# states the written operand; `mul!` on a `ROCArray` then reaches rocBLAS:
#     dispatch!(p, mul!, (use(p, C; write = true), use(p, A; read = true),
#                         use(p, B; read = true)))

# `register_backend!` in `__init__`, not `Mantle.initbackend!`, which belongs to
# the compiled-in backend (`src/vulkan/`) and would be replaced. Priority 50,
# below Vulkan (100) and Metal (90), so `Device()` keeps the platform backend;
# this one is requested as `Device(ROCmAPI())`.
function __init__()
    path = hipblaslt_library()
    if !isempty(path)
        lib = Libdl.dlopen(path; throw_error = false)
        if lib === nothing
            @warn "hipBLASLt at $path does not load; products with a bias run without it"
        else
            HIPBLASLT[] = lib
        end
    end
    register_backend!(; name = :rocm, priority = 50) do
        AMDGPU.functional() ? Mantle.Device(ROCmAPI()) : nothing
    end
end

end # module
