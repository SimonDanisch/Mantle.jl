"""
The host backend: the same graph, the same five analyses, executed on the CPU
through KernelAbstractions.

Compute only. There is no `render!`, no `Window` and no `Surface` here, because
there is nothing behind them — a missing method names what was unavailable,
which is what `Backend`'s docstring asks for.

Two things are deliberate and are the reason this is worth having rather than
being a stub:

**Transients share one arena.** They are views into a single `Vector{UInt8}` at
the offsets `Place` computed. Giving each its own `Array` would run, and would
make this backend structurally incapable of catching a placement bug — the one
thing it is best positioned to catch, because a wrong overlap here is wrong
bytes, deterministically, with no driver in the way.

**A step does no synchronisation.** `KernelAbstractions.synchronize(::CPU)` is a
no-op and a CPU launch has completed by the time it returns, so the scheduled
order IS the synchronisation. A barrier between every pass would make `Dag`,
`Schedule` and `Liveness` decorative.
"""
module MantleHostExt

using Mantle
using Mantle: Storage, BufferKind, Access
import KernelAbstractions as KA
import Mantle: storage, dispatch!, compute!, run!, free!

# ── device ────────────────────────────────────────────────────────────────────
struct HostDevice <: Mantle.Device
    backend::Any
    pool::Mantle.Pool          # the device owns it; nothing about it reaches the caller
end
HostDevice(backend) = HostDevice(backend, Mantle.Pool())
Mantle.pool(d::HostDevice) = d.pool
"""
    Device(Host())              # KA.CPU()
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
a KA launch, `render!` is a render pass, `custom!` is anything else. A model uses
`dispatch!` and never touches `render!` — same graph, same device, same pool as
an editor's chain. There is nothing for a second backend to add.

Dispatched on the backend TAG, never on the module: `MantleLavaExt` defines
`Device(::typeof(Lava))` and `typeof(Lava)` is `Module`, so a
`Device(::typeof(KernelAbstractions))` here would be the SAME signature and
whichever extension loaded second would silently replace the other.
"""
Mantle.Device(::Mantle.Host) = HostDevice(KA.CPU())
Mantle.Device(b::KA.CPU) = HostDevice(b)
Mantle.backend(d::HostDevice) = d.backend

# ── persistent resources ──────────────────────────────────────────────────────
#
# `HostBuffer`/`HostScalar` are gone: `Mantle.Buffer` and `Mantle.Scalar` are core
# types over pool regions, so a backend supplies four primitives and no type.

Mantle.rawalloc(::HostDevice, ::Mantle.Persistent, bytes::Int, c) = zeros(UInt8, max(bytes, 1))

"""Free physical memory. The honest bound for a host arena — and unlike the
device case there is swap behind it, so this is advice rather than a wall."""
Mantle.capacity(::HostDevice) = Int(min(Sys.free_memory(), UInt64(typemax(Int))))
Mantle.constraintof(::HostDevice, ::Mantle.Persistent, ts) = nothing

"The bytes of `a`, as a `T` view into its block. Host memory is directly addressable,
so this is a reinterpret rather than a mapping."
hostview(a::Mantle.DeviceArray{T}) where {T} =
    reinterpret(T, view(Mantle.memoryof(a), (Mantle.offset(a) + 1):(Mantle.offset(a) + sizeof(a))))

Mantle.deviceview(::HostDevice, a::Mantle.DeviceArray) = hostview(a)
Mantle.upload!(::HostDevice, a::Mantle.DeviceArray, first::Integer, data::AbstractVector) =
    (copyto!(hostview(a), first, data, 1, length(data)); a)
Mantle.download(::HostDevice, a::Mantle.DeviceArray) = collect(hostview(a))
Mantle.devicecopy!(::HostDevice, dst::Mantle.DeviceArray, src::Mantle.DeviceArray, n::Integer) =
    (copyto!(hostview(dst), 1, hostview(src), 1, n); dst)


# ── transients ────────────────────────────────────────────────────────────────
"""
A transient. `first`/`last` are filled in by `Liveness` from the scheduled order;
`view` stays `nothing` until `Place` hands it a slice of the arena, so a
transient that was never placed cannot be read by accident.
"""
mutable struct HostTransient{T} <: Mantle.Resource
    n::Int
    first::Int
    last::Int
    # `Any`, and NOT `Union{Nothing,Vector{T}}`. `materialize!` assigns a
    # `ReinterpretArray` over a view of the arena; that is not a `Vector`, so a
    # declared field type makes the assignment `convert` — which COPIES, and
    # every transient silently gets private storage. The arena then stays
    # untouched while every value still reads back correctly, so nothing fails
    # except the one property this backend exists to have.
    view::Any
end
storage(t::HostTransient) = t.view
Mantle.nbytes(t::HostTransient{T}) where {T} = t.n * sizeof(T)
# 64 bytes, not Vulkan's 256: a host arena has no device alignment requirement,
# and one cache line is what keeps two aliased transients off each other's lines.
Mantle.alignment(::HostTransient) = 64
Mantle.describe(t::HostTransient{T}) where {T} = "Transient.Buffer($T, $(t.n))"

"""One arena. Unlike Vulkan there is no buffer/image split to keep apart, so
every transient shares it and `Place` runs one placement."""
struct HostArena end
Mantle.arena(::HostTransient) = HostArena()

# ── graph ─────────────────────────────────────────────────────────────────────
mutable struct HostPass
    name::String
    usages::Vector{Pair{Int,Type}}
    dispatches::Vector{Any}
end
HostPass(name) = HostPass(String(name), Pair{Int,Type}[], Any[])
Mantle.usages(p::HostPass) = p.usages

mutable struct HostGraph <: Mantle.Graph
    dev::HostDevice
    passes::Vector{HostPass}
    transients::Vector{HostTransient}
    ids::IdDict{Any,Int}
    transient_by_id::Dict{Int,HostTransient}
    updates::Vector{Any}
    next::Int
end
Mantle.Graph(dev::HostDevice) =
    HostGraph(dev, HostPass[], HostTransient[], IdDict{Any,Int}(),
              Dict{Int,HostTransient}(), Any[], 0)

resourceid(g::HostGraph, r) = get!(g.ids, r) do
    g.next += 1
end

function Mantle.Transient.Buffer(g::HostGraph, ::Type{T}, n::Integer) where {T}
    t = HostTransient{T}(Int(n), typemax(Int), 0, nothing)
    push!(g.transients, t)
    g.transient_by_id[resourceid(g, t)] = t
    return t
end

struct PassHandle
    graph::HostGraph
    pass::HostPass
end

function Mantle.use(p::PassHandle, x; read::Bool = false, write::Bool = false)
    read || write || throw(ArgumentError("use() needs read, write, or both"))
    push!(p.pass.usages, resourceid(p.graph, x) => Storage{BufferKind,Access{read,write}})
    return x
end

struct Dispatch
    kernel::Any
    args::Tuple
    ndrange::Any
    group::Any
end

function compute!(f, g::HostGraph, name::AbstractString)
    p = HostPass(name)
    push!(g.passes, p)
    f(PassHandle(g, p))
    return p
end

dispatch!(p::PassHandle, kernel, args, ndrange; group = nothing) =
    push!(p.pass.dispatches, Dispatch(kernel, args, ndrange, group))

function Mantle.custom!(f, g::HostGraph, name::AbstractString)
    p = HostPass(name)
    push!(g.passes, p)
    body = f(PassHandle(g, p))
    applicable(body) || throw(ArgumentError(
        "custom!: the block has to return a zero-argument callable — it is what " *
        "runs at record time. Declare the uses, then return the work."))
    push!(p.dispatches, body)
    return p
end

# ── updates ───────────────────────────────────────────────────────────────────
"""
What `Update` hands back here. The same contract as the Lava one: calling it
stores a reference and touches nothing, and the write happens at the position
the graph reserved.

`pending` is atomic for the same reason — the caller may be whatever task set an
observable.
"""
mutable struct HostUpdateRef
    resource::Any
    range::Union{Nothing,UnitRange{Int}}
    @atomic pending::Any
end

(r::HostUpdateRef)(data) = (@atomic r.pending = data; nothing)

"""The single pass every `Update` shares, so one position in the schedule covers
them all. Declared `CopyDst` per resource, which is what orders it against the
passes that read them."""
function updates_pass!(g::HostGraph)
    for p in g.passes
        p.name == "updates" && return p
    end
    p = HostPass("updates")
    pushfirst!(g.passes, p)
    push!(p.dispatches, () -> writeupdates!(g))
    return p
end

"""
Apply whatever was fired since the last run.

An in-place copy, and no rename: host memory is directly addressable, so there
is no staging buffer to recycle and no hazard to schedule around — which also
means a resource's storage keeps its identity across a write, where the Lava
route replaces it.
"""
function writeupdates!(g::HostGraph)
    for r in g.updates
        data = @atomic r.pending
        data === nothing && continue
        writeupdate!(r.resource, r.range, data)
        @atomic r.pending = nothing
    end
    return nothing
end

writeupdate!(b::Mantle.Buffer, ::Nothing, data::AbstractVector) = Mantle.update!(b, data)
writeupdate!(b::Mantle.Buffer, r::AbstractUnitRange, data::AbstractVector) =
    Mantle.update!(b, r, data)
writeupdate!(s::Mantle.Scalar, ::Nothing, x) = Mantle.update!(s, x)

function Mantle.Update(g::HostGraph, buf; range = nothing)
    p = updates_pass!(g)
    id = resourceid(g, buf)
    any(u -> u.first == id, p.usages) || push!(p.usages, id => CopyDst)
    r = HostUpdateRef(buf, range, nothing)
    push!(g.updates, r)
    return r
end

# ── the compilation context ───────────────────────────────────────────────────
mutable struct Compile
    graph::HostGraph
    alias::Bool
    policy::Mantle.Policy
    analysis::Mantle.Analysis
    steps::Vector{Any}
end
Compile(g::HostGraph; alias = true, policy = Mantle.Overlap()) =
    Compile(g, alias, policy, Mantle.Analysis(), Any[])

# The thirteen methods core's phases ask of a context. That this list is short
# and dull is the point of the lift.
Mantle.analysis(c::Compile) = c.analysis
Mantle.passes(c::Compile) = c.graph.passes
Mantle.policy(c::Compile) = c.policy
Mantle.alias(c::Compile) = c.alias
Mantle.transients(c::Compile) = c.graph.transients
Mantle.transientbyid(c::Compile) = c.graph.transient_by_id
# No slices on this backend, so two ids name the same bytes exactly when they are
# the same id — which the `overlapping` docstring says is the floor.
Mantle.overlapping(::Compile, a::Int, b::Int) = a == b
Mantle.device(c::Compile) = c.graph.dev
Mantle.pool(c::Compile) = Mantle.pool(c.graph.dev)

# The four primitives core asks of a backend. No policy here: which block, what
# offset, when to grow and when to release are all decided in `Mantle.Pool`.
Mantle.rawalloc(d::HostDevice, ::HostArena, bytes::Int, constraint) =
    zeros(UInt8, max(bytes, 1))
Mantle.rawfree(::HostDevice, mem) = nothing
# Host memory is not VRAM: a 64 MiB block would fault in pages nobody asked for,
# and an allocation here is cheap enough that a small block is the right trade.
Mantle.blocksize(::HostDevice) = 1 << 20
Mantle.constraintof(::HostDevice, ::HostArena, ts) = nothing   # host memory is host memory
Mantle.compatible(::HostDevice, a, b) = true

"""
Give a transient its slice of the arena.

`reinterpret` over a `view`, so the bytes really are shared: two transients the
placer put at overlapping offsets will corrupt each other here, loudly and
deterministically. That is the property a separate `Array` per transient would
throw away.
"""
function Mantle.materialize!(t::HostTransient{T}, slab::Vector{UInt8}, offset::Int) where {T}
    n = t.n * sizeof(T)
    t.view = reinterpret(T, view(slab, (offset + 1):(offset + n)))
    return t
end

# ── barriers and pipelines: nothing to do, said explicitly ────────────────────
"""Nothing to emit. See `needs_transition(::Host, …)` — the scheduled order is
the synchronisation on this backend."""
Mantle.run!(::Mantle.Barriers, c::Compile) = c

"""
Bake the steps.

LAST, and that is a constraint rather than a convention: `resolve` reads a
transient's storage, which does not exist until `Place` has given it an offset.
Everything a frame could have decided is decided here — the kernel is
specialised, the arguments are bound, the ndrange is fixed — so `run!` is a loop
over callables and nothing else.
"""
function Mantle.run!(::Mantle.Pipelines, c::Compile)
    for p in Mantle.ordered(c), d in p.dispatches
        push!(c.steps, step(c, d))
    end
    return c
end

resolve(::Compile, x) = storage(x)

"""
One launch, fully resolved. `K`, `A` and `N` are concrete, so the call inside is
direct: no lookup, no splat over an abstract tuple, no `resolve`.
"""
struct Launch{K,A<:Tuple,N}
    kernel::K
    args::A
    ndrange::N
end
(l::Launch)() = l.kernel(l.args...; ndrange = l.ndrange)

kernelfor(k, ::Nothing, backend) = k(backend)
kernelfor(k, group, backend) = k(backend, group)

# A function barrier: `args` is concrete inside `Launch`, which is what makes the
# call in `(l::Launch)()` specialise.
step(c::Compile, d::Dispatch) =
    Launch(kernelfor(d.kernel, d.group, Mantle.backend(c.graph.dev)),
           map(a -> resolve(c, a), d.args), d.ndrange)
step(::Compile, body) = body        # a custom! body already IS the callable

# ── plan ──────────────────────────────────────────────────────────────────────
struct HostPlan <: Mantle.Plan
    graph::HostGraph
    steps::Vector{Any}
    arena::Vector{UInt8}
    peak::Int
    naive::Int
    regions::Vector{Any}       # the SHARED region of each arena — not owned
    arenas::Vector{Any}
    offsets::Vector{Int}
end

"""Give up this plan's claim on the arenas it was placed into. Explicit — see
`Mantle.trim!`. Deregistering rather than releasing, because the bytes are shared
with every other plan placed there."""
function Mantle.free!(pl::HostPlan)
    for ar in pl.arenas
        Mantle.untenant!(Mantle.pool(pl.graph.dev), ar, pl)
    end
    empty!(pl.regions); empty!(pl.arenas)
    return nothing
end

"""Re-materialise this plan's transients of `kind` into the arena's new region."""
function Mantle.remap!(pl::HostPlan, kind, region)
    for (i, t) in enumerate(pl.graph.transients)
        Mantle.arena(t) == kind || continue
        Mantle.materialize!(t, Mantle.memoryof(region), Mantle.offset(region) + pl.offsets[i])
    end
    return pl
end

function Mantle.Plan(g::HostGraph; alias = true, policy = Mantle.Overlap())
    c = Mantle.compile!(Compile(g; alias, policy))
    a = Mantle.analysis(c)
    arena = isempty(a.regions) ? UInt8[] : Mantle.memoryof(first(a.regions))
    pl = HostPlan(g, c.steps, arena, a.peak, a.naive, a.regions, a.arenas, a.offsets)
    for ar in a.arenas
        Mantle.tenant!(Mantle.pool(g.dev), ar, pl)
    end
    return pl
end

Mantle.peakbytes(pl::HostPlan) = pl.peak
naivebytes(pl::HostPlan) = pl.naive

"""
One frame: call the baked steps in the order compile settled on.

No synchronisation, no lookup, no branch. Everything that could be decided was.
"""
function run!(pl::HostPlan)
    for s in pl.steps
        s()
    end
    return nothing
end

end
