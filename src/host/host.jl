# The host backend: the same graph, the same five analyses, executed on the CPU
# through KernelAbstractions.
#
# Compute only. There is no `render!`, no `Window` and no `Surface` here, because
# there is nothing behind them — a missing method names what was unavailable,
# which is what `Backend`'s docstring asks for.
#
# Two things are deliberate and are the reason this is worth having rather than
# being a stub:
#
# **Transients share one arena.** They are views into a single `Vector{UInt8}` at
# the offsets `Place` computed. Giving each its own `Array` would run, and would
# make this backend structurally incapable of catching a placement bug — the one
# thing it is best positioned to catch, because a wrong overlap here is wrong
# bytes, deterministically, with no driver in the way.
#
# **A step does no synchronisation.** `KernelAbstractions.synchronize(::CPU)` is a
# no-op and a CPU launch has completed by the time it returns, so the scheduled
# order IS the synchronisation. A barrier between every pass would make `Dag`,
# `Schedule` and `Liveness` decorative.
# Included into `Mantle` rather than loaded as an extension. It was
# `ext/MantleHostExt.jl` triggered on KernelAbstractions, but KA is a hard
# dependency of core — `gemv.jl`, `fft.jl` and `narrow_phase.jl` all write
# `@kernel` — so the trigger always fired and the extension was one in name
# only. Worse, naming KA in both `[deps]` and `[weakdeps]` made Pkg record it as
# weak-only in the manifest, and `using KernelAbstractions` inside Mantle then
# failed outright.
#
# So it sits beside the other backends: `src/host/`, `src/vulkan/`, `src/metal/`.
# This one needs no weak dependency, so it needs no extension.

import KernelAbstractions as KA

# ── device ────────────────────────────────────────────────────────────────────
struct HostDevice <: Mantle.Device
    backend::Any
    pool::Mantle.Pool          # the device owns it; nothing about it reaches the caller
    accesses::Mantle.AccessCache
end
HostDevice(backend) = HostDevice(backend, Mantle.Pool(), Mantle.AccessCache())
Mantle.accesscache(d::HostDevice) = d.accesses
Mantle.pool(d::HostDevice) = d.pool
"""
    Device(HostAPI())              # KA.CPU()
    Device(KA.CPU())
    Device(LavaBackend())       # or CUDABackend, MetalBackend, …

**One Mantle device per physical device.** This one is for KA backends that have
no Mantle backend of their own — the CPU today.

It deliberately does NOT accept a `LavaBackend`. A draft did, on the theory that
a KA workload wants "scheduling and allocation, not render passes and barriers",
which is true and beside the point: it would hand `DNNKernels` a second
`Mantle.Device` with a second `Pool` sitting beside the real `LavaDevice` over one
VkDevice. Two pools, no sharing — the two-paths problem again, one level up, and
the sharing is the entire reason any of this exists.

**Fidelity is per PASS, not per backend, and it already exists.** `dispatch!` is
a KA launch, `render!` is a render pass, `trace!` is a hardware trace. A model
uses `dispatch!` and never touches `render!` — same graph, same device, same pool
as an editor's chain. There is nothing for a second backend to add.

Dispatched on the backend TAG, never on the module: `typeof(SomeModule)` is
`Module` for every module, so `Device(::typeof(Lava))` and
`Device(::typeof(KernelAbstractions))` would be the SAME signature,
and whichever loaded second would silently replace the other. Both are markers
now, `HostAPI()` and `VulkanAPI()`, which makes them different methods by
construction.
"""
Mantle.Device(::Mantle.HostAPI) = HostDevice(KA.CPU())
Mantle.Device(b::KA.CPU) = HostDevice(b)
Mantle.backend(d::HostDevice) = d.backend

# The host has no subgroup or cooperative-matrix hardware to advertise.  It
# still answers the portable capability query so graph clients can make the
# same feature decisions for every Mantle device.  A one-lane subgroup is the
# scalar CPU model; zero means that no bounded device-local memory budget or
# GPU occupancy count is available.
Mantle.caps(::HostDevice) = Mantle.DeviceCaps(false, 0, 1, 1, 0, 1024, 0, 0)

# Nothing is ever outstanding here: a CPU launch has completed by the time it
# returns, which is the same fact the header states about a step doing no
# synchronisation. Present so that `waitidle(device)` is answerable on every
# backend — core declares it and portable callers spell it that way, so leaving
# it out makes the host backend the one where the portable spelling is a
# `MethodError`.
Mantle.waitidle(::HostDevice) = nothing

# ── persistent resources ──────────────────────────────────────────────────────
#
# `Mantle.Buffer` and `Mantle.GPURef` are core types over pool regions, so a
# backend supplies four primitives and no type of its own.

Mantle.rawalloc(::HostDevice, ::Mantle.Persistent, bytes::Int, c) = zeros(UInt8, max(bytes, 1))

# An element type cannot ask host memory for anything it does not already have —
# there are no usage flags to get wrong, so every `T` gets the same `Vector{UInt8}`.
# Present because `persistentarray` asks every device this, and the host backend
# not answering made an ordinary `Buffer(host, zeros(Int32, 1))` a `MethodError`.
Mantle.bufferusage(::HostDevice, ::Type) = nothing

"""Free physical memory. The honest bound for a host arena — and unlike the
device case there is swap behind it, so this is advice rather than a wall."""
Mantle.capacity(::HostDevice) = Int(min(Sys.free_memory(), UInt64(typemax(Int))))
Mantle.constraintof(::HostDevice, ::Mantle.Persistent, ts) = nothing

# Nothing is ever in flight here: `run!` is a loop over callables and has
# returned by the time anyone could ask. So a fence carries no information, and
# `reclaim!` releases in the same call that stamps — the boundary it waits for on
# a device backend has already happened by construction.
Mantle.fence(::HostDevice) = UInt64(0)
Mantle.passed(::HostDevice, _) = true
# Already true, so there is never anything to wait for.
Mantle.waitfor(::HostDevice, _) = true

# A slab IS the host address space, so this backend's whole answer is a pointer
# and a length; `hostview`, `upload!`, `download` and `devicecopy!` are core's
# over it (see `hostspan` in `memory/resources.jl`), so neither this backend nor
# the Metal one spells that walk a second time.
#
# The bounds check it carries is worth naming once more, because this backend is
# the one every other is checked against: two transients the placer put at
# overlapping offsets corrupt each other loudly and deterministically, instead of
# each quietly getting private storage.
Mantle.hostspan(::HostDevice, slab::Vector{UInt8}) = (pointer(slab), length(slab))

Mantle.deviceview(d::HostDevice, a::Mantle.DeviceArray) = Mantle.hostview(d, a)
Mantle.upload!(d::HostDevice, a::Mantle.DeviceArray, first::Integer, data::AbstractVector) =
    Mantle.hostupload!(d, a, first, data)
Mantle.download(d::HostDevice, a::Mantle.DeviceArray) = Mantle.hostdownload(d, a)
Mantle.devicecopy!(d::HostDevice, dst::Mantle.DeviceArray, src::Mantle.DeviceArray, n::Integer) =
    Mantle.hostdevicecopy!(d, dst, src, n)


# ── transients ────────────────────────────────────────────────────────────────
#
# `TransientBuffer` is Mantle's, shared with every other backend. It was
# `HostTransient` here and `TransientBuffer` in the Vulkan backend: the same four
# fields under two names. What differs is only what `block` holds once `Place`
# has run — a device allocation there, a view over the host arena here.

# 64 bytes, not Vulkan's 256: a host arena has no device alignment requirement,
# and one cache line is what keeps two aliased transients off each other's lines.
Mantle.alignment(::HostDevice, ::Mantle.TransientBuffer) = 64

# `nbytes`, `describe` and `arena` are Mantle's — `arena(::TransientBuffer)`
# already answers `Buffers()`, which is the one arena this backend has. There is
# no buffer/image split to keep apart here, so `Place` runs one placement.

# ── graph ─────────────────────────────────────────────────────────────────────
#
# There is no graph here. `Graph`, `Pass`, `PassHandle`, `Compile` and `Plan` are
# Mantle's, and so are `newpass`, `use`, `passes`, `handle`, `dispatches` and
# `Update`, as are `Graph`, `Transient.Buffer`, `Plan` and `overlapping`. The
# ones below are what actually differs on a CPU.
# Every one of them was identical here and in the Vulkan backend, or differed
# only in a Vulkan-specific field that became a hook — see `makeprofiler` and
# `makeargmemory` in `graph/backend.jl`.

# ── the compilation context ───────────────────────────────────────────────────
#
# `Compile` is Mantle's too. Only the three answers below differ.

# No slices on this backend, so two ids name the same bytes exactly when they are
# the same id — which the `overlapping` docstring says is the floor.

# The four primitives core asks of a backend. No policy here: which block, what
# offset, when to grow and when to release are all decided in `Mantle.Pool`.
Mantle.rawalloc(d::HostDevice, ::Mantle.Buffers, bytes::Int, constraint) =
    zeros(UInt8, max(bytes, 1))
Mantle.rawfree(::HostDevice, mem) = nothing
# Host memory IS its own device memory, so the atom a move asks for is the slab's
# own pointer. Answered rather than defaulted: core has no fallback for this, on
# purpose — a backend that cannot say where its memory is cannot have a recorded
# plan patched when it moves, and silence would look like success.
Mantle.deviceaddress(::HostDevice, mem::Vector{UInt8}) = UInt64(pointer(mem))
# Host memory is not VRAM: a 64 MiB block would fault in pages nobody asked for,
# and an allocation here is cheap enough that a small block is the right trade.
Mantle.blocksize(::HostDevice) = 1 << 20
Mantle.constraintof(::HostDevice, ::Mantle.Buffers, ts) = nothing   # host memory is host memory
Mantle.compatible(::HostDevice, a, b) = true

"""
Give a transient its slice of the arena.

A wrapper over the arena's bytes and not a copy of them, so the bytes really are
shared — see [`hostview`](@ref) for why it is a wrapper and not a reinterpret.
The view goes in `block`, which is the field every backend keeps its placed
storage in.
"""
# The whole of what placement asks here: core's `materialize!` decides the
# extents, the element type and the offset, and this says what a host array over
# them is. `hostview` bounds-checks against the slab, which is why it takes its
# length rather than trusting the offset.
Mantle.deviceslice(::HostDevice, ::Type{T}, dims::Dims, slab::Vector{UInt8},
                   offset::Int) where {T} =
    Mantle.hostview(T, pointer(slab), length(slab), offset, dims)

# ── what differs on a CPU ─────────────────────────────────────────────────────
#
# The whole execution path — `Launch`, `bake`, `ndrangeof`, `DeferredRange`,
# `kernelfor`, the `Pipelines` phase and `run!(::Plan)` — is Mantle's, in
# `graph/kalaunch.jl`. Every KernelAbstractions backend runs a plan the same
# way. What is left below is the two answers a CPU gives differently.

"""Nothing to emit. See `needs_transition(::HostAPI, …)` — the scheduled order is
the synchronisation on this backend."""
Mantle.run!(::Mantle.Barriers, c::Mantle.Compile{HostDevice}) = c


# ── plan ──────────────────────────────────────────────────────────────────────
#
# `Plan` is Mantle's. `free!`, `remap!`, `peakbytes` and `naivebytes` are too —
# they read fields the same way on every backend. What is here is how a CPU
# builds one and how it runs it.


# `run!(::Plan)` is Mantle's, in `graph/kalaunch.jl`. This file carried a
# byte-identical copy of it specialised on `HostDevice` — which the collapse of
# host.jl onto the shared graph missed, and which then silently shadowed the
# shared one. That is not a theoretical cost: the shared method grew `Update`
# handling and this copy did not, so every `Update` on the host backend wrote
# nothing at all.

"""
Nothing to wait for.

The host backend's "device work" runs on the calling thread, so a count an
earlier step wrote is already visible. Overriding this is what keeps the shared
`ndrangeof` free here while it synchronises on a GPU — see `graph/kalaunch.jl`.
"""
Mantle.awaitwrites(::HostDevice) = nothing

"""
`true`: a call runs here, because this backend WALKS its plans.

`openrecording` answers `nothing`, so `run!` executes the plan's compiled
dispatches in order on this thread — and a call is one of them. There is no
command buffer for the call's work to be missing from.
"""
Mantle.runscalls(::HostDevice) = true

"""
Yes — a CPU has `Float64`, and saying so is not redundant.

There are TWO functions spelled `supports_float64` here. `KernelAbstractions` defines
one over its own `KernelAbstractions.Backend`, and `KernelInterface` defines another
over `KernelInterface.Backend`; the hierarchies are separate, and `KA.CPU` belongs
only to the first. Everything in this repo asks the **KernelInterface** one — Mantle
answers it for the Metal backend, and `DNNKernels`' `narrowfusedconsts` and `emit`
call it as `KI.supports_float64` — so `KA.CPU` reached it with no method at all and
`Model(graphs, weights; backend = CPU())` threw a `MethodError` out of a constant
narrowing pass.

Stated for `CPU` and not for `KA.Backend` at large: KernelInterface's contract is that
a backend implements this only when the answer is NO, so a blanket `true` over that
hierarchy would make a GPU that lacks `Float64` and forgot to declare silently claim
it. A CPU having `Float64` is a fact; the rest is the other backends' to say.
"""
Mantle.KI.supports_float64(::KA.CPU) = true

"""
Yes, trivially: nothing here is recorded. A GPU discards a predicated iteration
with conditional rendering because its commands are already written; this
backend interprets the plan on every run, so "discard" is `withpredicate`
reading the flag and not calling the body — which it can do because the flag is a host
array the gate kernel just wrote on this very thread.
"""
Mantle.supportspredicate(::HostDevice) = true
Mantle.syncbackend(::HostDevice) = Mantle.HostAPI()

# ── what a kernel does to its arguments ───────────────────────────────────────
#
# Nothing is adapted on the way in — a launch here passes `resolve` itself — so
# the default `devicetype` is already right and only the transient case is
# stated, as the `Array` `materialize!` wraps the slab in.

Mantle.devicebuffertype(::HostDevice, ::Type{T}, N::Int) where {T} = Array{T,N}

# On a CPU every array IS device memory, and `materialize!` hands a transient a
# plain `Array` over the arena's bytes. A `StructArray` is not one: it is a
# container of them, and the walk opens it up like any other struct.
Mantle.isdevicearray(::Array) = true

"""
The same walk, through Julia's own method table.

There is no overlay here and no adaptation: a CPU kernel IS the Julia function,
run on this thread, over the arrays `resolve` produced. `nothing` for the
interpreter is what says so.
"""
Mantle.kernelinterpreter(::HostDevice, kernel, tt) = nothing

function Mantle.kakernelaccesssignature(d::HostDevice, kernel, argT::Tuple, ndrange, group)
    be = Mantle.backend(d)
    obj = Mantle.kernelfor(kernel, group, be)
    ndr, _ws, iterspace, dynamic = KA.launch_config(obj, Mantle.dispatchrange(ndrange), nothing)
    # The same five-argument `mkcontext` a CPU launch calls per block — the block
    # index does not change the context's TYPE, which is all the walk needs, but
    # the arity does: `Kernel{CPU}` has no three-argument method.
    block = first(KA.NDIteration.blocks(iterspace))
    ctx = KA.mkcontext(obj, block, ndr, iterspace, dynamic)
    tt  = (typeof(ctx), argT...)
    return nothing, obj.f, tt
end
