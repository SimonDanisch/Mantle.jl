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
end
HostDevice(backend) = HostDevice(backend, Mantle.Pool())
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

Dispatched on the backend TAG, never on the module. The Vulkan backend used to
define `Device(::typeof(Lava))`, and `typeof(Lava)` is `Module` — so a
`Device(::typeof(KernelAbstractions))` here would have been the SAME signature,
and whichever loaded second would silently replace the other. Both are markers
now, `HostAPI()` and `VulkanAPI()`, which makes them different methods by
construction.
"""
Mantle.Device(::Mantle.HostAPI) = HostDevice(KA.CPU())
Mantle.Device(b::KA.CPU) = HostDevice(b)
Mantle.backend(d::HostDevice) = d.backend

# Nothing is ever outstanding here: a CPU launch has completed by the time it
# returns, which is the same fact the header states about a step doing no
# synchronisation. Present so that `waitidle(device)` is answerable on every
# backend — core declares it and portable callers spell it that way, so leaving
# it out makes the host backend the one where the portable spelling is a
# `MethodError`.
Mantle.waitidle(::HostDevice) = nothing

# ── persistent resources ──────────────────────────────────────────────────────
#
# `HostBuffer`/`HostScalar` are gone: `Mantle.Buffer` and `Mantle.GPURef` are core
# types over pool regions, so a backend supplies four primitives and no type.

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
Mantle.fence(::HostDevice) = nothing
Mantle.passed(::HostDevice, _) = true
# Already true, so there is never anything to wait for.
Mantle.waitfor(::HostDevice, _) = true

# A slab IS the host address space, so this backend's whole answer is a pointer
# and a length; `hostview`, `upload!`, `download` and `devicecopy!` are core's
# over it (see `hostspan` in `memory/resources.jl`). What used to be `wrapbytes`
# here, and `hostview` again in the Metal backend, is one function now.
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
# `Update`. This backend used to define all of them a second time; the ones below
# are what actually differs on a CPU.
# `Graph`, `Transient.Buffer`, `Plan` and `overlapping` are all Mantle's now.
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
function Mantle.materialize!(t::Mantle.TransientBuffer{T}, slab::Vector{UInt8},
                             offset::Int) where {T}
    t.block = Mantle.hostview(T, pointer(slab), length(slab), offset, (t.n,))
    t.offset = offset
    return t
end

# ── what differs on a CPU ─────────────────────────────────────────────────────
#
# The whole execution path — `Launch`, `bake`, `ndrangeof`, `DeferredRange`,
# `kernelfor`, the `Pipelines` phase and `run!(::Plan)` — is Mantle's, in
# `graph/kalaunch.jl`. Every KernelAbstractions backend runs a plan the same
# way, and this file used to be the only copy of it. What is left below is the
# two answers a CPU gives differently.

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
Yes, trivially: nothing here is recorded. A GPU discards a predicated iteration
with conditional rendering because its commands are already written; this
backend interprets the plan on every run, so "discard" is `withpredicate`
reading the flag and not calling the body — which it can do because the flag is a host
array the gate kernel just wrote on this very thread.
"""
Mantle.supportspredicate(::HostDevice) = true
Mantle.syncbackend(::HostDevice) = Mantle.HostAPI()
