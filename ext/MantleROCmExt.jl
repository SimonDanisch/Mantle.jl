"""
Mantle's ROCm backend: the graph, the allocator and the plan on AMDGPU.jl.

**Verbs only.** Every piece of Mantle that DECIDES anything — `Dag`, `Schedule`,
`Liveness`, `Place`, `Aliasing`, `Barriers`, `Pipelines`, the pool, `record!`,
`run!`, `execute!` — is core's, and nothing below replaces any of it. What is
here is the list of questions a HIP device answers differently: the allocator
primitives, the four transfer verbs, a timeline, what a dispatch compiles to,
the four verbs that open and close a recording and a run, and the capability
table. `src/metal/` fills in the same list, and `graph/backend.jl` is where it
is declared.

Overriding `Mantle.recordplan!` and `Mantle.execute!` — core's orchestration —
is what this backend does NOT do, and neither does it carry a copy of core's run
body or a `WeakKeyDict` state machine to decide when to capture: that state
machine works around a problem that disappears once kernels are compiled in
[`compile_dispatch`](@ref), which is the verb Mantle already has for exactly
that.

**Written against the merged `sd/lava-refactor`** (`be9e117`), which matters for
one method in particular. That commit deleted `resource_moved!`/`arena_moved!`
in favour of [`Mantle.deviceaddress`](@ref) — "a hook that can be a silent
no-op and still look implemented is the shape of that bug" — and a version of
this backend written before the merge implemented exactly those two. On the
merged tree that is not an overload at all: it DEFINES two functions nothing
calls, so a move leaves the captured graph reading freed storage with nothing
said, which is the bug the deletion was for. What replaces them here is
`deviceaddress` plus [`Mantle.patchable`](@ref), and the audit that found it is
worth keeping: every `Mantle.x` this file names has to exist in Mantle, because
a name that does not is a new definition rather than a method.

**What it is FOR**, as against running the same kernels through plain AMDGPU.jl:

  * *Allocation.* One `hipMalloc`-class block per arena with the graph's
    transients suballocated into it at the offsets `Place`/`Aliasing` computed,
    instead of one `ROCArray` per intermediate and HIP's pool underneath.
  * *Libraries.* [`deviceview`](@ref) hands back a genuine borrowed `ROCArray`,
    so rocBLAS, MIOpen, rocFFT and any `ROCArray` code read a Mantle region with
    no copy and no knowledge that it is one. This is the thing the Vulkan
    backend cannot do at all: on Lava a GEMM is a SPIR-V kernel we wrote. On
    SAM 2.1 large's 195 encoder GEMMs, at the same fp32 accumulation, rocBLAS is
    2.3x Lava's cooperative-matrix kernel (~42 ms against ~96 ms) — and about
    half of that is given back by the bias and activation Lava fuses into its
    store and `rocblas_gemm_ex` has no way to. `bench/rocm_vs_vulkan.jl` has
    both halves, and `LibOp` there is how a library call becomes a pass.

    SAM 2.1 large encode at 1024x1024, same GPU and same session: **322 ms here
    against 269 ms on Lava**, and the whole of `runsam2` 451 ms against 312.
    That gap was 647 ms when this backend was first written, and none of the
    three things that closed it were in the backend: the workgroup autotune this
    file was skipping (see [`hipcompile`](@ref)), and two predicates in
    `DNNKernels` that asked "is this Lava's array type" where they meant "is
    this a dense device array" — one of them gating the one-kernel layer norm,
    which cost 854 graph passes against Lava's 96.

    A GEMM library is not the whole of what a DNN runtime needs from a vendor
    stack. On gfx11 and gfx12, this backend now exposes AMDGPU's native 16x16
    WMMA lowering through the same backend-independent cooperative-matrix API,
    so DNNKernels emits its one-pass fused flash-attention kernel instead of the
    old score/softmax/apply sequence. Dense identity-activation `addmm`s use an
    optional hipBLASLt bias epilogue when `libhipblaslt` is available; exact
    GELU remains a separate DNNKernels kernel because the vendor epilogue is not
    the same function. rocBLAS remains the fallback library GEMM, and convolution
    still uses DNNKernels where no suitable vendor plan is integrated.

    SAM 2 is also the WORST case, and four models is a fairer picture than one.
    Steady state, same GPU, same session, this backend against Lava:

        DepthAnything  ViT + DPT, conv-heavy       85 ms   against  131 ms
        RIFE           optical flow, 1920x1152    162 ms   against  141 ms
        NeuralLUT      26 ops, 33^3 output       1.57 ms   against 2.00 ms
        SAM 2.1        encoder, 1082 passes       322 ms   against  269 ms

    Medians of fifteen. RIFE's Lava column read 519 ms when this was first
    written, with a spread out to 912, and that was a bug in the Vulkan pool's
    soft-cap collection rather than anything about the work — the collection was
    spaced by how OFTEN it may run and not by what it costs, so a workload 2%
    over the cap paid ~63 ms of GC every 20 ms forever. Fixed in
    `runtime/memory.jl` (`gc_budget`); the honest number is 141.

    Faster on two of the four, level on a third, and behind where
    cooperative-matrix attention carries the graph. All four agree with Lava on their outputs: the
    depth map to 0.07% of its mean, RIFE to six digits, NeuralLUT to six, and
    SAM 2's decoder EXACTLY. `runsam2` end to end scores 0.0404 here against
    0.0403 through Lava, differing on 6 pixels of 65,536.

    Two of those four are run-to-run non-reproducible on BOTH backends, which is
    not this backend's doing: a split-K convolution accumulates with float
    atomics and `conv_implicit.jl` says so in as many words. On ROCm the
    variation is one fp16 ULP on a few percent of pixels.

    The encoder's outputs agree with the PyTorch references to 0.0016 and 0.0071
    on the two feature maps and exactly on the three positional encodings, and
    `add_129` comes out at 1.398 against the 0.2..0.6 that `SAM2Runner`'s gate
    pins for Lava. That node is NOT this backend's: both Mantle backends first
    diverge from PyTorch at `add_82`, by 7.9% of its scale through Lava and
    10.5% through here — as measured while the layer norm was still the six-pass
    fallback here, and the one-kernel form took `add_129` from 4.244 to 1.398,
    so the share is smaller now. `add_129` is that divergence grown; see the
    comment on that gate for how it was localised. The DECODER is exact:
    `verifygraph` reports zero mismatches against the references here.
  * *Submit baking.* Through HIP graphs: a recording is a `hipGraph` captured
    from the stream, and a run is one `hipGraphLaunch` instead of one
    `hipModuleLaunchKernel` per dispatch. **Worth almost nothing in TIME
    once the dispatch is compiled at `Pipelines` time**, which is the honest
    number: 1.93 us a dispatch captured against 2.20 us walked, on 64 tiny
    dispatches.
    Before that move the walk cost 6.22 us and the graph looked like a 3.2x win
    — but what those 4 us were is KernelAbstractions re-deriving the iteration
    space and looking the kernel up per launch, not submission. Lava's recorded
    command buffer is still 0.98 us, so a HIP submit is about twice a Vulkan
    one and the graph does not close that. On a GPU-bound graph it is worth
    nothing at all: SAM 2.1's encoder replays in 322 ms and WALKS in the same
    322 ms, because 1,082 launches' host cost hides entirely behind the work.

    What it is worth is ALLOCATION. A walk reaches the allocator per launch —
    40,976 bytes for those 64 dispatches, 10.5 MB for one call of the encoder —
    and a replay is one `hipGraphLaunch` that touches no host memory at all.
    Zero bytes, measured on the 1,082-pass encoder plan and pinned in
    `test/rocm/test_rocm.jl`. A replay that allocates is a replay a garbage
    collection can interrupt while the previous launch is still in flight.

**Why an extension and not an `@static include`** like `src/vulkan/` and
`src/metal/`: AMDGPU.jl is a genuinely optional dependency rather than a
platform fact. A Linux machine gets the Vulkan backend compiled in whether or
not AMDGPU.jl is present, and this one loads beside it rather than instead of
it — `availablebackends()` answers `[:vulkan, :rocm]` here. That is why every
method below is written `Mantle.x`: a name missed in a module of its own does
not fail, it silently defines `MantleROCmExt.release!`, and nothing breaks until
the wrong one is called. `src/host/host.jl` is written the same way.
"""
module MantleROCmExt

import AMDGPU
import GPUArrays
import KernelAbstractions as KA
import KernelInterface as KI
import Mantle
import LinearAlgebra
import Libdl
using Mantle: Pool, Buffers, Images, Persistent, TransientBuffer, DeviceArray,
    DeviceInfo, ROCmAPI, register_backend!, selectdevice
using KernelInterface: DeviceCaps, MatrixShape

const Managed = AMDGPU.Managed
const HIPBuffer = AMDGPU.Mem.HIPBuffer

# ── device ────────────────────────────────────────────────────────────────────

"""
    Device(ROCmAPI())
    Device(AMDGPU.ROCBackend())

A HIP device: one `HIPDevice`, one stream, one pool, one timeline.

**The stream is created here and owned, and the device ADOPTS it** — every verb
below that touches the GPU makes it the task's current stream and does not put
back what was there. That follows the Metal backend, whose `defaultdevice!`
adopts its queue process-wide, and it is not a convenience.

AMDGPU.jl's stream is TASK-LOCAL, and two things break when work lands on one
the device is not using. The claim `needs_transition(::ROCmAPI, …)` rests on —
one in-order stream, so the scheduled order is the synchronisation — is false
across two streams: a plan whose transients the placer had overlapped, run from
two tasks, came back with 8.0 in the first quarter of a buffer and 9.0, 10.0 and
11.0 in the other three, with nothing raised and `waitidle` returning promptly
because it was synchronising a stream the kernels were not on. And a
`hipStreamBeginCapture` is invalidated by any synchronise on the capturing
stream, which is exactly what AMDGPU.jl does when it finds an argument whose
`Managed` was last touched somewhere else.

The timeline counts RETIREMENTS and not submissions, for the reason
`src/metal/device.jl` gives at length: this backend does not own the submission
of a `Launch` — AMDGPU.jl does — so it has nothing to signal an event from, and
a counter nothing advances makes `passed` answer `true` for memory the GPU is
still reading.
"""
mutable struct ROCmDevice <: Mantle.Device
    dev::AMDGPU.HIPDevice
    stream::AMDGPU.HIP.HIPStream
    # hipBLASLt's handle belongs to this concrete device.  It is deliberately
    # carried by the device and passed into calls, never recovered through an
    # ambient/global device cache.
    blaslt::Ptr{Cvoid}
    pool::Pool
    # `next` is the fence a retirement gets; `completed` is the highest value a
    # real synchronise has covered.
    next::UInt64
    completed::UInt64
    # Whether a wait is `hipStreamSynchronize` or AMDGPU.jl's polling one. See
    # `waitidle`.
    blocking::Bool
    caps::Union{Nothing,DeviceCaps}
    accesses::Mantle.AccessCache
end

const HIPBLASLT_LIB = Libdl.find_library(["libhipblaslt.so.1", "libhipblaslt.so"])

function hipblaslt_handle()
    isempty(HIPBLASLT_LIB) && return C_NULL
    h = Ref{Ptr{Cvoid}}(C_NULL)
    status = ccall((:hipblasLtCreate, HIPBLASLT_LIB), Cint,
                   (Ref{Ptr{Cvoid}},), h)
    return status == 0 ? h[] : C_NULL
end

function ROCmDevice(dev::AMDGPU.HIPDevice = AMDGPU.device())
    # On `dev` and not on whichever device happens to be current: `HIPStream()`
    # creates on the current one, so a device built for the second GPU of two
    # would otherwise get a stream belonging to the first.
    stream = dev == AMDGPU.device() ? AMDGPU.HIPStream() :
             AMDGPU.device!(AMDGPU.HIPStream, dev)
    d = ROCmDevice(dev, stream, C_NULL, Pool(), UInt64(0), UInt64(0), true, nothing,
                   Mantle.AccessCache())
    adopt!(d)
    d.blaslt = hipblaslt_handle()
    return d
end

"""Make this device's stream the current task's. Idempotent, and it does not put
back what was there — see the struct's docstring for why adopting is the point."""
@inline function adopt!(d::ROCmDevice)
    AMDGPU.stream() == d.stream || AMDGPU.stream!(d.stream)
    return d
end

# One device per process, cached: a second `ROCmDevice` is a second `Pool` over
# the same HIP device, which is two allocators over one memory. Same rule and
# same reason as `METAL_DEVICE`.
const ROCM_DEVICE = Ref{Union{Nothing,ROCmDevice}}(nothing)

function Mantle.Device(::ROCmAPI; select = nothing)
    if select === nothing
        d = ROCM_DEVICE[]
        d === nothing || return adopt!(d)
        s = get(ENV, "MANTLE_DEVICE", nothing)
        hip = s === nothing ? AMDGPU.device() :
              AMDGPU.devices()[selectdevice(s, Mantle.devices(ROCmAPI()))]
        return ROCM_DEVICE[] = ROCmDevice(hip)
    end
    return ROCmDevice(AMDGPU.devices()[selectdevice(select, Mantle.devices(ROCmAPI()))])
end

# `integrated` is a real HIP property and it is the one that matters for the
# selector's ranking: on this APU the "device" memory is system memory, which is
# why `capacity` below is 112 GiB and not a card's VRAM.
Mantle.devices(::ROCmAPI) =
    [DeviceInfo(i, AMDGPU.HIP.name(d),
                AMDGPU.HIP.properties(d).integrated == 1 ? :integrated : :discrete,
                "ROCm $(AMDGPU.HIP.runtime_version())")
     for (i, d) in enumerate(AMDGPU.devices())]

Mantle.defaultdevice!(d::ROCmDevice) = (ROCM_DEVICE[] = d; adopt!(d))
Mantle.Device(::AMDGPU.ROCBackend) = Mantle.Device(ROCmAPI())
Mantle.pool(d::ROCmDevice) = d.pool
Mantle.backend(::ROCmDevice) = AMDGPU.ROCBackend()
# TWO backend objects, and they are different types. `AMDGPU.ROCBackend` is
# KernelAbstractions', which is what a `@kernel` constructor takes;
# `ROCInterface.ROCBackend` is the KernelInterface one, which is what
# `kernel_function`, `argconvert` and `auto_launch_sizes` are defined on. Core
# asks for the second by name instead of assuming `backend(dev)` answers both.
Mantle.kibackend(::ROCmDevice) = AMDGPU.ROCInterface.ROCBackend()
# Stated for BOTH backend objects above, because callers reach this through
# either one. KernelInterface's `= true` default is declared on `KI.Backend`
# and neither of these subtypes it, so without these a Float64 literal reaching
# `DNNKernels.kernelnumber` threw `MethodError` and no graph could be built.
KI.supports_float64(::AMDGPU.ROCBackend) = true
KI.supports_float64(::AMDGPU.ROCInterface.ROCBackend) = true
Mantle.accesscache(d::ROCmDevice) = d.accesses
Mantle.devicebuffertype(::ROCmDevice, ::Type{T}, N::Int) where {T} =
    AMDGPU.Device.ROCDeviceArray{T,N,1}
Mantle.argtype(d::ROCmDevice, y) =
    Core.Typeof(KI.argconvert(Mantle.kibackend(d), y))
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

Wait for everything submitted to this device's stream.

**`blocking = true`, and it is worth 1.3 ms a call.** AMDGPU.jl's default is a
non-blocking synchronise that polls through the Julia scheduler, and the latency
that adds does not show up on an idle stream — 0.033 ms either way with nothing
outstanding — only when there is work to wait for. Measured on the 64-dispatch
chain in `bench/rocm_vs_vulkan.jl`, one run plus one wait:

    blocking = true      0.141 ms
    blocking = false     1.421 ms

A run is 0.14 ms of device work, so the polling wait costs ten times the frame.
It is a field rather than a constant because the non-blocking path exists for a
reason: a kernel using device-side printf or dynamic allocation is served by a
HOSTCALL, which needs the host to be in the scheduler to answer it, and a
blocking wait on such a kernel can deadlock. Nothing in Mantle's own vocabulary
emits one, so the fast wait is the default; a caller whose kernels do says
`dev.blocking = false`.

That was not hypothetical on this stack, and finding it is what fixed it.
Running SAM 2.1's encoder here printed four of AMDGPU.jl's "Global hostcalls
detected" notices, every one a device-side allocation the kernels did not mean
to make: `safetrunc` was being dispatched DYNAMICALLY because its target type
travelled as a broadcast argument, and three float-to-integer conversions
carried an `InexactError` branch their own guards had already made unreachable.
A throw in a kernel has to allocate its exception, and on a device that is a
hostcall. Fixed in `DNNKernels`' `ops.jl` and `kernels/resample.jl` — the
encoder now runs with no hostcall at all, which is what makes the fast wait
safe here rather than merely lucky.
"""
Mantle.waitidle(d::ROCmDevice) =
    (AMDGPU.synchronize(d.stream; blocking = d.blocking); nothing)

Mantle.awaitwrites(d::ROCmDevice) = Mantle.waitidle(d)

# ── allocation ───────────────────────────────────────────────────────────────
#
# HIP has no equivalent of Vulkan's buffer usage flags and no equivalent of
# Metal's storage modes that matters here: `hipMalloc` memory is bytes, and what
# a kernel does with them is decided when the pointer is passed. So a constraint
# is `nothing`, every block can host every request, and `mergeconstraints` has
# nothing to merge. The one distinction HIP does have — device versus pinned host
# memory — is not one the graph makes: an arena the CPU reads is served by
# `download` below, through the copy engine, exactly as on a discrete card.

Mantle.constraintof(::ROCmDevice, ::Union{Buffers,Images,Persistent}, ts) = nothing
Mantle.compatible(::ROCmDevice, blk, req) = true
Mantle.mergeconstraints(::ROCmDevice, kind, a, b) = nothing
Mantle.bufferusage(::ROCmDevice, ::Type) = nothing

"""
    rawalloc(dev, kind, bytes, constraint) -> Managed{HIPBuffer}

One device allocation. The pool suballocates within it and never asks for
another until this one cannot serve a request.

A `Managed` and not a bare `HIPBuffer`: `Managed` is what carries the stream a
buffer was last touched on, and `convert(Ptr, ::Managed)` is what synchronises
when that stream is not the current one. Handing out views over a bare buffer
would drop that, and the resulting hazard is a read of memory a copy on another
stream had not finished writing.

`max(bytes, 1)` because a zero-length `hipMalloc` gives back `C_NULL` and the
pool is entitled to ask for an empty block, the same guard the other three
backends carry.
"""
function Mantle.rawalloc(d::ROCmDevice, ::Union{Buffers,Images,Persistent},
                         bytes::Int, constraint)
    adopt!(d)
    buf = HIPBuffer(max(bytes, 1); stream = d.stream)
    buf.ptr == C_NULL && throw(OutOfMemoryError())
    return Managed(buf; stream = d.stream)
end

"""
Release a block's device memory, now rather than at the next GC.

Through AMDGPU.jl's own `pool_free`, which is the destructor a `ROCArray`'s
`DataRef` would have carried: it handles a stream that has since become invalid
by falling back to the default one, and it accounts the bytes back to
`memory_stats` so `AMDGPU.info()` and `Mantle.capacity` keep agreeing.
"""
Mantle.rawfree(d::ROCmDevice, mg::Managed) = (adopt!(d); AMDGPU.pool_free(mg); nothing)

"""64 MiB, as on Vulkan and Metal: one device allocation amortised over many
transients. Nothing about HIP argues for a different number."""
Mantle.blocksize(::ROCmDevice) = 64 << 20

"""What HIP will hand out right now, not what the device has. `hipMemGetInfo`'s
free figure is the honest bound for an arena the placer is checking against."""
Mantle.capacity(::ROCmDevice) = AMDGPU.free()
Mantle.maxalloc(d::ROCmDevice) = Int(AMDGPU.HIP.properties(d.dev).totalGlobalMem)

# ── the timeline ─────────────────────────────────────────────────────────────

function Mantle.fence(d::ROCmDevice)
    d.next += UInt64(1)
    return d.next
end

Mantle.passed(d::ROCmDevice, f) = UInt64(f) <= d.completed

"""
Wait until the timeline reaches `f`.

A full stream synchronise, because this backend cannot observe the submission
that reads a region — see the struct's docstring. `issued` is read BEFORE the
wait so the value recorded is the one the wait actually covered; reading it
after would credit the sync with retirements made while it was blocked.
"""
function Mantle.waitfor(d::ROCmDevice, f)
    Mantle.passed(d, f) && return true
    issued = d.next
    Mantle.waitidle(d)
    d.completed = max(d.completed, issued)
    return true
end

# ── regions as arrays ────────────────────────────────────────────────────────

"""
    rocview(mg, T, dims, byteoffset) -> ROCArray{T}

A **borrowed** `ROCArray` over `dims` elements of `T` at `byteoffset` in a pool
block. No copy, and no second owner: the `DataRef`'s destructor frees nothing,
because the `HIPBuffer` belongs to the pool and a block that two owners free is
freed out from under every other tenant of it. `AMDGPU.unsafe_wrap` builds the
same shape for the same reason.

**`ROCArray`'s `offset` field is checked and not assumed.** Its unit differs
between versions: AMDGPU 2.8.0 as installed says "offset of the data in memory,
in bytes", and the 2.2.1 checkout beside it says "Offset is in number of
elements (not bytes)". Dividing by `sizeof(T)` for the second placed every view
at a quarter of its intended offset on the first, which is not an error
anywhere — the chain in `bench/rocm_vs_vulkan.jl` came back holding 8.0, 9.0,
10.0 and 11.0 in the four quarters of a buffer that should have been all 8.0,
because the views overlapped each other rather than the slots the placer chose.
So the address is computed independently and compared, and a unit flip is a
throw at the first view instead of arithmetic nobody checks.
"""
function rocview(mg::Managed, ::Type{T}, dims::Dims{N}, byteoffset::Int) where {T,N}
    need = prod(dims) * sizeof(T)
    bytes = Int(sizeof(mg))
    byteoffset + need <= bytes || throw(ArgumentError(
        "$(join(dims, "x")) $T at offset $byteoffset needs $need bytes, block has $bytes"))
    # Not a driver requirement but an arithmetic one: a `T` load from an address
    # that does not divide `sizeof(T)` is misaligned. `alignment` below and
    # `persistentarray`'s 256 both guarantee it, so this is a claim about the
    # allocator rather than a recoverable case.
    byteoffset % sizeof(T) == 0 || throw(ArgumentError(
        "offset $byteoffset does not divide sizeof($T) = $(sizeof(T))"))
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

The bridge this backend exists for: rocBLAS, MIOpen, rocFFT and any `ROCArray`
code operate on a Mantle region without copying it and without knowing it is
one.
"""
Mantle.deviceview(::ROCmDevice, a::DeviceArray{T,N}) where {T,N} =
    rocview(Mantle.memoryof(a)::Managed, T, size(a), Mantle.offset(a))

"""
Give a placed transient its slice of the arena.

`block` holds the borrowed `ROCArray` a launch will pass, built once here rather
than per launch — core's `storage(::TransientBuffer)` hands it straight back.
"""
Mantle.deviceslice(::ROCmDevice, ::Type{T}, dims::Dims, slab::Managed,
                   offset::Int) where {T} = rocview(slab, T, dims, offset)

"""
256 bytes.

HIP does not publish a `minStorageBufferOffsetAlignment`, so the number is not a
driver requirement; it is what keeps two aliased transients out of each other's
cache lines (128 bytes on this architecture) and keeps every element type's
natural alignment satisfied so `rocview`'s arithmetic is exact.
"""
Mantle.alignment(::ROCmDevice, ::TransientBuffer) = 256

# `Mantle.usagekey(a::ROCArray) = a.buf[]` was here, deleted 2026-09-15. It
# keyed a hazard on the pool block behind an array, because two `deviceview`s of
# one region are different `ROCArray` objects and an intercepted launch had only
# the object. A declared pass names the transient, which is one object.

# ── moving bytes ─────────────────────────────────────────────────────────────
#
# `hostspan` is deliberately NOT implemented. It is the hook for a backend whose
# device memory the CPU can address directly, and `hipMalloc` memory is not
# mapped into the process address space even on this APU, where it is physically
# the same DRAM. Answering it with a device pointer would hand `hostview` an
# address that segfaults on first touch. So the three transfer verbs are written
# here, over `copyto!`, which is `hipMemcpyAsync` on the device's own stream.

function Mantle.upload!(d::ROCmDevice, a::DeviceArray, first::Integer,
                        data::AbstractVector)
    adopt!(d)
    copyto!(Mantle.deviceview(d, a), Int(first), data, 1, length(data))
    return a
end

"""
    download(dev, a) -> Vector

`a`'s contents on the host. Synchronises first, for the reason core's
`hostdownload` does: a copy queued behind a kernel that is still running returns
what was there before it.
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
One dispatch of the graph, compiled: the HIP kernel, the iteration space it was
compiled for, and the grid it launches with.

Everything `KernelAbstractions` would work out per launch is worked out once, at
`Pipelines` time. That is not only cheaper — it is what makes a HIP graph
capturable at all. A kernel compiled inside a `hipStreamBeginCapture` invalidates
it (`hipErrorStreamCaptureInvalidated`), and every kernel here is compiled by
GPUCompiler on its first launch, so a capture taken over a walk that still had
compiling to do would fail.

`MetalRecordedDispatch` is the same idea in `src/metal/record.jl`, down to being
callable so that core's `emitdispatch!(::Immediate, d, name)` runs it unchanged.
"""
struct ROCmCompiledDispatch{K,C,A<:Tuple,R<:Tuple,G,N}
    kernel::K
    ctx::C
    args::A
    # The graph resources `args` were resolved FROM, kept so `openrecording` can
    # notice that one of them has moved since. See `checkresolved`.
    raw::R
    # An `Int` or a tuple of up to three, and the difference is the difference
    # between the two paths below.
    #
    # A `@kernel` reads its own index out of the `CompilerMetadata` in `ctx`, so
    # its geometry is carried there and the LAUNCH can be flat: `length` over
    # the iteration space is what `ROCKernels` itself passes.
    #
    # A macro-free kernel has no such context. It asks the hardware directly,
    # with `get_group_id().y`, so the launch geometry IS the geometry, and
    # flattening it made every multi-dimensional dispatch silently
    # one-dimensional: the implicit-GEMM convolution puts its NPQ block and its
    # split index on the second grid axis, so every workgroup read block 1 and
    # the output came back wrong by the magnitude of its own values.
    groupsize::G
    gridsize::N
end

function (d::ROCmCompiledDispatch)()
    # A grid of zero launches nothing, which is what it means: a wavefront round
    # whose queue emptied is the ordinary case rather than an edge. Core's
    # `Launch` makes the same check.
    prod(d.gridsize) == 0 && return nothing
    # `ctx === nothing` is the KernelInterface form: a KI kernel is a plain
    # function, so there is no leading `CompilerMetadata` to pass. The branch is
    # on a field whose type is known — `Nothing` or the metadata — so it
    # specialises away.
    if d.ctx === nothing
        d.kernel(d.args...; groupsize = d.groupsize, gridsize = d.gridsize)
    else
        d.kernel(d.ctx, d.args...; groupsize = d.groupsize, gridsize = d.gridsize)
    end
    return nothing
end

# Zero bytes of argument memory and no indirect slot: a launch passes its
# arguments directly, and this backend records a `DeviceRange`'s ceiling rather
# than narrowing a command from a device-written count. Same answers as
# `MetalRecordedDispatch`, for the same reasons.
Mantle.argsize(::ROCmCompiledDispatch) = 0
Mantle.indirectindex(::ROCmCompiledDispatch) = 0
Mantle.devicesized(::ROCmCompiledDispatch) = false

# ── a CALL ───────────────────────────────────────────────────────────────────
#
# A call is core's `Mantle.Call`, and there is no `ROCmCallDispatch`: that would
# be this with an `ndrange` keyword bolted on, for a convention where a callable
# kernel is launched as `body(args...; ndrange)`, and it is unreachable with
# `buildskernel` routing every non-`@kernel` to the KernelInterface path.
#
# Nothing about a call is this backend's, which is why the type is not:
# `compile_dispatch` hands a call to core's `bake` and what comes back runs
# `f(args...)`. What IS this backend's is that a call can be RECORDED here,
# because a recording is a stream capture: rocBLAS submits kernels to the
# current stream and nothing else, so the capture gets them. A body that
# synchronises or reads back would invalidate the capture, and that is a
# property of the body rather than something this backend can check —
# `closerecording!` reports it if it happens.
#
# `openrecording` below refuses a plan holding a `Mantle.Launch`, which is a
# different thing: an ndrange whose count only exists on the device, read on the
# host, which really cannot be captured. A `Call` is not a `Launch`.

"""
    kicompile(compile, dev, dispatch) -> ROCmCompiledDispatch

Compile a macro-free kernel through `KernelInterface`.

`Mantle.kikernel` reaches `KI.kernel_function`, which is `hipfunction` here, and
`KI.auto_launch_sizes` is the launch configuration. Both happen HERE and neither
happens per run: a capture cannot contain host work, and a replay has none to do.

**Not 40 lines of this file doing it by hand.** Reading whether a workgroup size
was requested off the value `KA.launch_config` RETURNED is wrong, because that
call substitutes a
default of its own for a `DynamicSize` kernel — so the occupancy autotune never
ran, and 1,014 of SAM 2.1's broadcast kernels replayed in the wrong group shape
for 29% of the encoder (647.3 ms recorded against 454.9 eager; 460.8 after).
`auto_launch_sizes` is the same three steps done once, upstream, where every
backend gets them: `kernel_max_work_group_size` for the occupancy limit,
`threads_to_workgroupsize` to SHAPE it to the ndrange — the step whose absence
was the bug — then `cld.(ndrange, workgroupsize)`.
"""
function kicompile(c::Mantle.Compile{ROCmDevice}, dev::ROCmDevice, d::Mantle.Dispatch)
    be = Mantle.kibackend(dev)
    args = map(a -> Mantle.resolve(dev, a), d.args)
    nd = Mantle.dispatchrange(d.ndrange)
    ndt = nd isa Integer ? (Int(nd),) : map(Int, Tuple(nd))
    kern = Mantle.kikernel(d.kernel, dev, args)
    wgin = d.group === nothing ? () :
           (d.group isa Integer ? (Int(d.group),) : map(Int, Tuple(d.group)))
    ng, wg = KI.auto_launch_sizes(kern, (), wgin, ndt)
    # The tuples, not their products: see `ROCmCompiledDispatch`. This is the
    # same call `AMDGPU`'s own `KI.Kernel` launch makes, and it takes up to
    # three dimensions on either.
    return ROCmCompiledDispatch(kern.kern, nothing, args, d.args, wg, ng)
end

"""
    compile_dispatch(compile, dispatch, argoff, indirect)

Compile one dispatch into something the walk can run and a capture can record.

One kind is handed back to core's `bake` instead, as a plain `Launch`: a dispatch
over a `DeviceRange` with no ceiling, whose count only exists on the device.
Core's interpreted launch reads that count on the HOST, which is a device
synchronise — legal in a walk and illegal inside a capture. `openrecording`
refuses a plan holding one rather than recording something that would be
invalidated.
"""
function Mantle.compile_dispatch(c::Mantle.Compile{ROCmDevice}, d::Mantle.Dispatch,
                                 argoff::Int, indirect::Int)
    dev = c.graph.dev
    d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing &&
        return Mantle.bake(c, d)
    # A CALL first: no ndrange, so core's `bake` resolves the arguments and
    # hands back a `Mantle.Call`. Capturable here, unlike on a backend that
    # builds a command buffer — see the note above this function's neighbours.
    Mantle.iscall(d) && return Mantle.bake(c, d)
    # KERNELINTERFACE next, for a kernel that is a plain function. Asked of
    # `backend(dev)`, the KA backend, because that is what a `@kernel`
    # constructor takes; `kicompile` then compiles against `ROCInterface`'s,
    # which is the KI one and a different type.
    Mantle.buildskernel(d.kernel, Mantle.backend(dev)) || return kicompile(c, dev, d)
    obj = Mantle.kernelfor(d.kernel, d.group, Mantle.backend(dev))
    isempty(fieldnames(typeof(obj.f))) ||
        throw(ArgumentError("dispatch!: the kernel closes over $(fieldnames(typeof(obj.f))). " *
                            "A plan resolves its arguments once, so a captured device array " *
                            "would be the one it had when the plan was built — pass it as a " *
                            "dispatch argument instead, where a rename is followed."))
    args = map(a -> Mantle.resolve(dev, a), d.args)
    nd = Mantle.dispatchrange(d.ndrange)
    # `callgroup`, not `nothing`: a kernel whose workgroup is `DynamicSize` was
    # launched with `workgroupsize = wg` and carries the size nowhere else, so
    # dropping it here sends the autotune below to pick a different one — and a
    # kernel body that reads its own workgroup size from a `Val` argument then
    # strides by a number the dispatch did not use.
    group = Mantle.callgroup(obj, d.group)
    ndrange, _, iterspace, _ = KA.launch_config(obj, nd, group)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel, ctx, iterspace = hipcompile(obj, args, ndrange, group, ctx, iterspace)
    return ROCmCompiledDispatch(kernel, ctx, args, d.args,
                                length(KA.workitems(iterspace)),
                                length(KA.blocks(iterspace)))
end

"""
The HIP kernel for this body and these argument types, and the iteration space it
ends up compiled for.

`hipfunction` rather than `@roc launch = false`, which is the macro over it. The
type tuple is built from the CONVERTED arguments because that is what the
kernel's parameters are — `rocconvert` is what turns a `ROCArray` into the
`ROCDeviceArray` a kernel sees.

The second half is the autotune `ROCKernels` does per launch, against the kernel
that will actually run: a body whose workgroup size is dynamic is partitioned by
the occupancy its own code allows, and the context has to be rebuilt for that
partition. Checked rather than assumed that the repartition leaves the context's
TYPE alone — a kernel compiled for one context type and launched with another's
bytes reads its ndrange out of the wrong fields.

`group` is the size the CALLER asked for and NOT the one `KA.launch_config`
handed back, which is the same distinction `ROCKernels` makes and for the same
reason. Given `nothing` for a `DynamicSize` kernel, `launch_config` substitutes
a default of its own — `min(prod(ndrange), _max_group_size)` in the first
dimension — and returns THAT, so a condition reading the returned size never
sees `nothing` and this autotune never ran. It cost 29% of SAM 2.1's encoder:
1,014 of its 1,840 passes are broadcast kernels launched without a workgroup
size, and every one of them replayed in the default group where an eager launch
had used what the occupancy allows. 647.3 ms recorded against 454.9 ms eager,
which is what made a baked plan look slower than no plan at all; 460.8 ms after,
against 457.4 ms eager.
"""
function hipcompile(obj, args, ndrange, group, ctx, iterspace)
    tt = hiptypes(ctx, args)
    kernel = AMDGPU.hipfunction(obj.f, tt)
    if KA.workgroupsize(obj) <: KA.DynamicSize && group === nothing
        (; groupsize) = AMDGPU.launch_configuration(kernel)
        ws = AMDGPU.ROCKernels.threads_to_workgroupsize(groupsize, ndrange)
        iterspace, _ = KA.partition(obj, ndrange, ws)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
        tt2 = hiptypes(ctx, args)
        tt2 === tt || (kernel = AMDGPU.hipfunction(obj.f, tt2))
    end
    return kernel, ctx, iterspace
end

hiptypes(ctx, args) =
    Base.to_tuple_type(map(Core.Typeof, map(AMDGPU.rocconvert, (ctx, args...))))

# ── a recording, and a run ───────────────────────────────────────────────────
#
# A recording here is a `hipGraph`: the stream is put into capture mode, the
# plan's own walk submits into it, and what comes out is an executable graph.
# That is why the emitter `openrecording` hands back is core's `Immediate` one
# rather than a recorder of this backend's — under capture, "do it now" IS "write
# it down", which is the whole mechanism HIP graphs provide. Nothing about the
# walk has to know it is being recorded, so there is no second pass over the plan
# and no `emitdispatch!`/`emitbarriers!` of our own.

"""
`true`: this backend records plans, and every plan it accepts is recordable.

The Metal backend answers `false` because its `openrecording` may DECLINE a plan
and `run!` then has to walk it rather than refuse it. Nothing here declines —
[`Mantle.openrecording`](@ref) either records or throws — so `true` is the
honest answer, and it is the one a caller asking "can I record on this backend"
needs: `DNNKernels.recordplan` reads exactly this before it builds a graph, and
with `false` it refuses to record at all.
"""
Mantle.recordsplans(::ROCmDevice) = true

"""
`true`: a call runs here, and records.

Because a recording is a stream CAPTURE. The call runs on the host during the
capture, rocBLAS submits its kernels to the capturing stream, and those are
exactly what the graph ends up holding — so the replay is the library's work
without the library being involved again. A backend that builds a command buffer
itself cannot say this; see `runscalls`' docstring in `graph/backend.jl`.

A body that synchronises or reads back would invalidate the capture. That is a
property of the body rather than of calls, and `closerecording!` reports it.
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
        ltcheck(ccall((:hipblasLtMatmulDescCreate, HIPBLASLT_LIB), Cint,
                      (Ref{Ptr{Cvoid}}, Cint, Cint), desc, 2, 0), "descriptor creation")
        epilogue = Ref{UInt32}(4) # HIPBLASLT_EPILOGUE_BIAS
        ltcheck(ccall((:hipblasLtMatmulDescSetAttribute, HIPBLASLT_LIB), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 2, epilogue, sizeof(UInt32)), "bias epilogue")
        biasptr = Ref{Ptr{Cvoid}}(Ptr{Cvoid}(pointer(bias)))
        ltcheck(ccall((:hipblasLtMatmulDescSetAttribute, HIPBLASLT_LIB), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 3, biasptr, sizeof(Ptr{Cvoid})), "bias pointer")
        biastype = Ref{Int32}(2)
        ltcheck(ccall((:hipblasLtMatmulDescSetAttribute, HIPBLASLT_LIB), Cint,
                      (Ptr{Cvoid}, Cint, Ptr{Cvoid}, Csize_t),
                      desc[], 4, biastype, sizeof(Int32)), "bias type")
        for (layout, rows, cols, ld) in
                ((ad, M, K, M), (bd, K, N, K), (dd, M, N, M))
            ltcheck(ccall((:hipblasLtMatrixLayoutCreate, HIPBLASLT_LIB), Cint,
                          (Ref{Ptr{Cvoid}}, Cint, UInt64, UInt64, Int64),
                          layout, 2, rows, cols, ld), "matrix layout")
        end

        alpha = Ref{Float32}(1)
        beta = Ref{Float32}(0)
        GC.@preserve D A B bias begin
            ltcheck(ccall((:hipblasLtMatmul, HIPBLASLT_LIB), Cint,
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
            layout == C_NULL || ccall((:hipblasLtMatrixLayoutDestroy, HIPBLASLT_LIB),
                                      Cint, (Ptr{Cvoid},), layout)
        end
        desc[] == C_NULL || ccall((:hipblasLtMatmulDescDestroy, HIPBLASLT_LIB),
                                  Cint, (Ptr{Cvoid},), desc[])
    end
    return D
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

Put this device's stream into capture mode, so the walk of `passes` is written
into a graph instead of executed. `passes` is the whole plan unless `record!` was
asked for a partition, in which case this is called once per piece and each
piece becomes its own graph.

This took `(dev, plan)` until 2026-09-26, after core's signature had gained
`passes` on 2026-09-22 (`2dc58eb`). The two-argument method was then a new
function nothing called: core's default answered "declined", every ROCm plan was
walked kernel by kernel instead of replayed as a graph, and asking for a
partition threw. `test/rocm/test_rocm.jl` pins the signature.

**A library a recorded plan will call has to be initialised before this.**
rocBLAS creates its handle on first use, and creating one while the stream is
capturing segfaults inside `rocblas_create_handle` — not an error this can
check for, and not one it can recover from. In the flow this backend was built
for it costs nothing: `DNNKernels.call` runs the graph once immediately and
records afterwards, so every library it touches is already up.

Throws rather than declining, for a plan holding a dispatch over a `DeviceRange`
with no ceiling. Core's interpreted launch reads that count on the host, and a
host read on a capturing stream invalidates the capture — so there is no graph to
be had, and saying so is better than handing back a plan that looks recorded.
Such a plan needs `emitkernel!`, which this backend does not implement; see
`supportspredicate` for the other half of the same gap.
"""
function Mantle.openrecording(d::ROCmDevice, pl::Mantle.Plan, passes::AbstractUnitRange)
    checkresolved(d, pl)
    piece = view(pl.passes, passes)
    any(pp -> any(x -> x isa Mantle.Launch, pp.dispatches), piece) && throw(ArgumentError(
        "record!: this plan dispatches over a `DeviceRange` with no ceiling, whose " *
        "count is read on the host. A capture is invalidated by a host read, so " *
        "there is no graph to record — give the range a `max`, or run the plan " *
        "without recording it."))
    adopt!(d)
    # ── warm every CALL before the capture opens ─────────────────────────────
    #
    # `rocblas_create_handle` inside a capture SEGFAULTS in libamdhip64. Not
    # "invalidates the capture", which `closerecording!` would report: a signal
    # 11 through `rocblas_create_handle` -> `library_state` -> `gemm!`, because
    # rocBLAS creates its handle lazily on first use and handle creation
    # allocates on the capturing stream.
    #
    # So each call runs once here, against the arguments the plan resolved: the
    # library initialises, compiles what it compiles and takes its workspace,
    # and the capture that follows sees only the kernels it submits. This is the
    # same warm-up HIP's own graph documentation asks for, and it is this
    # backend's business rather than a caller's precondition — the alternative
    # is a crash rather than an error.
    #
    # One extra execution per call per `record!`. Its writes land in plan-owned
    # bytes that the replay writes again, so the only thing it changes is that
    # the library is ready. A GATED pass's call runs here too, which is work the
    # gate might have skipped; the gate is still evaluated per run inside the
    # graph, so what this costs is one execution and not a wrong answer.
    # Only this piece's passes: a partitioned record opens one capture per
    # piece, and warming the whole plan each time would run every call once
    # per piece.
    for pp in piece, cd in pp.dispatches
        cd isa Mantle.Call && cd()
    end
    # Resolve AMDGPU's stream ownership before capture as well as its lazy
    # library state. `hiptypes` converted these arguments while compiling, but
    # the same array may subsequently be used by another stream (constant
    # folding does this across several short plans). The next conversion would
    # then synchronize its previous owner; inside capture HIP rejects that as
    # `hipErrorStreamCaptureUnsupported`. Conversion here performs any transfer
    # while it is legal, and the captured launch sees the already-current owner.
    for pp in piece, cd in pp.dispatches
        cd isa ROCmCompiledDispatch || continue
        # The one-argument form is for reflection: its adaptor deliberately
        # carries `stream = nothing` and therefore does NOT take ownership.
        # Use the launch stream explicitly; otherwise this loop is a no-op and
        # the HIPKernel wrapper attempts the transfer after capture has begun.
        map(arg -> AMDGPU.rocconvert(arg, d.stream), cd.args)
    end
    # Everything already submitted has to land before the capture opens: a
    # capture records the stream, and work in flight on it is work the graph
    # would wait for once and never again. The warm-up above is included in
    # that, which is why it goes first.
    Mantle.waitidle(d)
    AMDGPU.HIP.begincapture!(d.stream)
    return Mantle.Immediate()
end

"""
    closerecording!(emitter, plan) -> HIPGraphExec

Close the capture and instantiate what it recorded.
"""
function Mantle.closerecording!(::Mantle.Immediate, pl::Mantle.Plan{ROCmDevice})
    d = pl.graph.dev
    graph = endcapture!(d)
    graph === nothing && throw(ErrorException(
        "record!: the capture was invalidated, so there is no graph. Something " *
        "in the walk reached the host — a synchronise, an allocation on the " *
        "capturing stream, or a kernel that still had to be compiled."))
    return ROCmRecording(AMDGPU.HIP.instantiate(graph), graph)
end

"""
A plan's recording: the executable graph, and the graph it came from.

The second field is only there to be REACHABLE. `HIPGraph` carries a finalizer
that calls `hipGraphDestroy`, and instantiating is documented to make the exec
independent of it — but keeping the graph alive for the life of the exec costs a
field and removes the question.
"""
struct ROCmRecording{E,G}
    exec::E
    graph::G
end

"""
    release!(recording)

Nothing to do, and not for want of a destroy call.

`hipGraphExecDestroy` and `hipGraphDestroy` exist, and both handles already carry
an unguarded `finalizer` that calls them — so destroying here would free the same
graph twice, once now and once when the GC gets round to the object. The bytes go
back when the last reference does.

That makes it the one place this backend answers a lifetime verb with a no-op,
and it is worth saying why rather than leaving the method absent: `invalidate!`
calls this whenever a move throws a recording away, and an absent method is a
`MethodError` from inside the pool's lock — which is where it first showed.
"""
Mantle.release!(::ROCmRecording) = nothing

"""
    abandonrun!(dev, emitter)

A walk that threw still has to close its capture. A stream left capturing fails
every later `hipStreamSynchronize` with `hipErrorStreamCaptureUnsupported`, which
is how one failure inside a walk turns into a device that appears broken.
"""
Mantle.abandonrun!(d::ROCmDevice, ::Mantle.Immediate) = (endcapture!(d); nothing)

"""End the capture if there is one, and give back the graph or `nothing` when it
was invalidated. AMDGPU's `endcapture!` reads the status rather than throwing,
which is what lets this be called while unwinding from an error."""
endcapture!(d::ROCmDevice) =
    AMDGPU.HIP.is_capturing(d.stream) ? AMDGPU.HIP.endcapture!(d.stream) : nothing

"""
    checkresolved(dev, plan)

Refuse to record a plan whose resources have moved since it was compiled.

`compile_dispatch` resolves each argument ONCE, at `Pipelines` time, and a
compiled dispatch holds the resulting `ROCArray` — so a `resize!` that relocates
a persistent buffer leaves it holding the vacated storage. Core invalidates the
RECORDING on a move (see [`Mantle.patchable`](@ref)) and `run!` records again,
but re-recording walks the same compiled dispatches and captures the same stale
address.

The Vulkan backend does not have this problem and that is not luck: its compiled
command holds a pointer INTO the plan's argument memory rather than the array,
so patching those eight bytes is enough. Rebuilding a compiled dispatch here
would mean writing into `PassPlan.dispatches`, which is core's structure and
core's business; what belongs to a backend is noticing and saying so. A plan
whose resources moved has to be built again.

Costs one `resolve` per argument per `record!`, which happens once, and nothing
per run.
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

The emitter for whatever this run has to say, and the point at which the device's
stream becomes the task's. Every launch the walk makes reads `AMDGPU.stream()`,
so adopting here is what puts a run of this plan on the stream `waitidle` and the
capture use.
"""
Mantle.openrun(d::ROCmDevice, pl::Mantle.Plan) = (adopt!(d); Mantle.Immediate())

"""
    closerun!(dev, plan, emitter)

Submit the run: the plan's recording, if it has one.

Two methods rather than one with an `Any` third argument, because core's own
`closerun!(dev, pl, ::Immediate)` is as specific in its third argument as a
`Union` would be — one method covering both is ambiguous with it rather than more
specific. Core hands `nothing` for a run that had nothing to say in front of its
recording.
"""
Mantle.closerun!(d::ROCmDevice, pl::Mantle.Plan, ::Mantle.Immediate) = submitrun!(d, pl)
Mantle.closerun!(d::ROCmDevice, pl::Mantle.Plan, ::Nothing) = submitrun!(d, pl)

function submitrun!(d::ROCmDevice, pl::Mantle.Plan)
    rec = pl.recording
    # `submitrecording!` and not a launch inline, because the recording may be a
    # `RecordingParts` — several graphs from a partitioned `record!` — and
    # iterating those is core's, not this backend's.
    rec === nothing || Mantle.submitrecording!(d, rec, nothing)
    # The token `waitfor!(plan)` waits on. This backend's timeline counts what it
    # has issued rather than what the GPU has signalled, so waiting on it is a
    # full synchronise — which is what `waitfor` already does with it.
    return Mantle.fence(d)
end

"""
    submitrecording!(dev, recording, emitter)

Launch one captured graph on this device's stream.

The third argument is the run's own front emitter, and there is never one here:
this backend's `openrun` hands back core's `Immediate`, so whatever a run had to
say of its own already went to the stream before this is reached. On Vulkan it is
a one-shot command buffer that has to be submitted together with the first piece.

Several of these in order IS a partitioned recording: HIP has no equivalent of
one submission carrying several command buffers, so the pieces are separate
`hipGraphLaunch`es on the same in-order stream, which is the same ordering and
the same completion points.
"""
function Mantle.submitrecording!(d::ROCmDevice, rec::ROCmRecording, _)
    adopt!(d)
    AMDGPU.HIP.launch(rec.exec, d.stream)
    return nothing
end

"""A capture that will not be closed into a graph, because the walk threw. The
stream has to come out of capture mode either way: one left capturing fails every
later `hipStreamSynchronize` with `hipErrorStreamCaptureUnsupported`. Same body as
[`Mantle.abandonrun!`](@ref) and a separate method because they are separate
verbs — see core's docstring for why."""
Mantle.abandonrecording!(d::ROCmDevice, ::Mantle.Immediate) = (endcapture!(d); nothing)

"""
    deviceaddress(dev, mg) -> UInt64

Where this allocation begins in the GPU's address space.

The whole of what a move asks a backend for — core works out which region moved,
by how much, and which recorded plans hold an address inside it. A `Managed`'s
`HIPBuffer` is a device pointer, so this is that pointer.
"""
Mantle.deviceaddress(::ROCmDevice, mg::Managed) =
    UInt64(convert(Ptr{UInt8}, mg.mem))

"""
`false`: a captured graph holds its kernel arguments where nothing can rewrite
them.

`hipGraphExecKernelNodeSetParams` would need the node handles, and
`hipStreamEndCapture` does not hand them back — so there is no patch target, and
core throws the recording away on a move instead. It costs one walk at a
`resize!` or an arena growth and nothing per run.

There are no `resource_moved!`/`arena_moved!` methods here walking the pool's
own listener list and invalidating by hand. Core declares no such hooks, so
defining them here defines two functions nothing calls, and a move then leaves
the graph reading freed storage with nothing said: the exact bug the hooks would
be there to prevent.
"""
Mantle.patchable(::ROCmDevice) = false

"""
No, and not for the reason a recording backend says no.

Core's interpreted `withpredicate` discards an iteration by reading the flag on
the host — `storage(pred)[i + 1].go` — and that is a scalar index into a
`ROCArray`, which AMDGPU.jl refuses ("Scalar indexing is disallowed"). It would
also be a full stream synchronise per gated pass, since the gate kernel that
wrote the flag is queued. So a gated pass holding fixed-size work is refused at
graph build with the message `repeat!` prints, rather than at the first frame.

The other half of `repeat!` — a pass whose every dispatch is a `DeviceRange`, so
the fused prepare writes zero groups for a discarded iteration — needs
`emitkernel!`, which neither this backend nor the host nor Metal implements. So
`repeat!` does not work here yet; it is one method away, and that method is
shared with the other two.
"""
Mantle.supportspredicate(::ROCmDevice) = false

# ── capabilities ─────────────────────────────────────────────────────────────

"""
    caps(dev) -> DeviceCaps

What the portable kernels ask about this device. `gemv.jl` sizes its workgroup
from `workgrouplimit` and `fft.jl` its staging from `sharedbudget`.

On gfx11 and gfx12, `coopmat` reports AMDGPU's native WMMA lowering of
KernelInterface's portable matrix operations: one subgroup-scoped 16x16x16
Float16 product with Float32 accumulation.  No workgroup-scoped shape is
advertised.  Other architectures stay disabled until their own native fragment
adapter exists; a HIP device being present is not by itself a matrix-capability
claim.
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
        # cores: the field means "SMs / CUs", and HIP's count is not that on
        # RDNA. `multiProcessorCount` is 20 on this part where Vulkan's
        # `shaderCoreCount` is 40 and the part is specified as 40 CUs, because
        # ROCm counts WORKGROUP PROCESSORS here and a WGP is two CUs. Left as
        # HIP reports it: doubling it is an architecture-conditional (on CDNA
        # the count IS CUs) and it buys nothing measurable — DepthAnything, the
        # conv-heaviest model in tree and the one whose tile heuristics read
        # this, runs 91.7 ms at 20 and 92.6 ms at 40. Worth knowing if a tiling
        # surprise ever traces back here.
        Int(p.multiProcessorCount),
        # HIP exposes the maximum resident threads per processor directly.
        # DeviceCaps speaks in subgroups, so convert rather than throwing the
        # fact away.  Attention's shared-memory-heavy tile uses this to avoid
        # launching an eight-wave workgroup on a processor that can keep 64
        # waves resident; on gfx1151 the measured 16-wave form is 32-35% faster.
        Int(p.maxThreadsPerMultiProcessor) ÷ Int(p.warpSize),
        NTuple{4,Int}[],                    # wggran: no granularity table
        coopmat ? [MatrixShape(Float16, Float32, 16, 16, 16,
                               KI.SubgroupScope())] : MatrixShape[],
    )
end

Mantle.caps(b::AMDGPU.ROCBackend) = Mantle.caps(Mantle.Device(b))

# ── what a library submits, DECLARED ────────────────────────────────────────
#
# `recordhook`, `recordop!` (four methods), `HIPLaunch`, `BLASMul`, `retain` and
# `AMDGPU.RECORD_HOOK` were here, deleted 2026-09-15 with the capture path.
#
# They existed because a kernel library submits work this backend did not write:
# a bare `@roc` launch (`mapreduce`'s partials), a `rocBLAS` GEMM, a `hipMemset`
# behind `fill!`. With a capture open each had to be noticed and turned into a
# pass, which meant a hook inside AMDGPU.jl itself — about 200 lines of local
# patches to a package this one is not allowed to name.
#
# Declared, a library call needs no hook and no wrapper type either. It is the
# three-argument `dispatch!`, whose absence of an ndrange says "call this, do
# not launch it":
#
#     dispatch!(p, mul!, (use(p, C; write = true),
#                         use(p, A; read = true),
#                         use(p, B; read = true)))
#
# `BLASMul(tA, tB, α, β)` is not needed for the ordinary case: `mul!` on a
# `ROCArray` already reaches rocBLAS, so what the caller states is the thing no
# library announces — which operand is WRITTEN, the thing `useleaves!` could not
# know. The arguments live in the plan, so nothing has to be retained behind the
# library's back either (`retain` existed because `mapreduce` frees its own
# partials while the recording still names them).

# `register_backend!` in `__init__` and NOT `Mantle.initbackend!`: that method
# belongs to the backend compiled in at parse time (`src/vulkan/` here), and
# defining it again would REPLACE Vulkan's registration rather than add to it.
#
# Priority 50, below Vulkan's 100 and Metal's 90, so `Device()` and
# `defaultbackend()` keep answering with the platform's own backend. This one is
# asked for by name: `Device(ROCmAPI())`.
function __init__()
    register_backend!(; name = :rocm, priority = 50) do
        AMDGPU.functional() ? AMDGPU.ROCBackend() : nothing
    end
end

end # module
