# The graph's data structures, shared by every backend. Driver objects appear only
# as type parameters (`Graph{D}`, `PassPlan{D,B}`, `ArgMemory{S}`, `Profiler{Q}`,
# `TransientImage`); a backend records and submits what core compiles.

"""The swapchain image. Acquire returns whatever image the presentation engine
chooses, not `frame % n`, so `run!` brackets the acquire, wait and present."""
struct WindowSurface <: Resource
    win::Any
end

"""
What a draw re-reads every frame, as a cell the host may rewrite between runs: its
arguments, count, index buffer, instance count and texture bindings. A plan fixes
the pipeline, argument layout and draw kind. The type parameters keep
`CompiledDraw` concrete on the per-draw record path, so [`rebind!`](@ref) stores
only values of these types.
"""
mutable struct DrawBinding{A<:Tuple,C,I,B}
    args::A
    count::C
    indices::I
    instances::Int
    # Rebound like the rest: a growing glyph atlas is updated in place, so neither a
    # baked texture nor an `objectid` cache key sees the change.
    bindings::B
end

"""
    DrawBinding(device, args, count; indices = nothing, instances = 1, bindings = nothing)

A rebindable cell holding one draw's per-frame values, resolved for `device`.
"""
DrawBinding(dev, args::Tuple, count; indices = nothing, instances::Integer = 1,
            bindings = nothing) =
    DrawBinding(argmap(a -> resolve(dev, a), args), count,
                indices === nothing ? nothing : resolve(dev, indices), Int(instances),
                bindings)

"""
    rebind!(binding, device, args, count; indices = nothing, instances = 1,
            bindings = binding.bindings)

Store this frame's values in `binding` for the next run of every plan drawing from
it, without recompiling. Values are `convert`ed to the binding's types; one that
does not convert throws, since it means a different pipeline or draw kind.
"""
function rebind!(b::DrawBinding{A,C,I,B}, dev, args::Tuple, count;
                 indices = nothing, instances::Integer = 1,
                 bindings = b.bindings) where {A,C,I,B}
    b.args = convert(A, argmap(a -> resolve(dev, a), args))
    b.count = convert(C, count)
    b.indices = convert(I, indices === nothing ? nothing : resolve(dev, indices))
    b.instances = Int(instances)
    b.bindings = convert(B, bindings)
    return b
end

# Accessors over a plain tuple (a draw that re-reads nothing) or a `DrawBinding`.
@inline drawargs(t::Tuple) = t
@inline drawargs(b::DrawBinding) = b.args
@inline boundcount(::Tuple, c) = c
@inline boundcount(b::DrawBinding, _) = b.count
@inline boundindices(::Tuple, i) = i
@inline boundindices(b::DrawBinding, _) = b.indices
@inline boundinstances(::Tuple, n) = n
@inline boundinstances(b::DrawBinding, _) = b.instances
@inline boundbindings(::Tuple, x) = x
@inline boundbindings(b::DrawBinding, _) = b.bindings

struct DrawCall
    shader::Any
    # A `Tuple`, or a [`DrawBinding`](@ref) cell the host rewrites between runs.
    args::Any
    count::Any
    frag_args::Tuple
    # `nothing` takes the pass's viewport. Per draw because a plotting frame draws
    # many differently clipped rectangles into one pass; declared rather than set
    # between draws, so the scheduler may reorder passes.
    viewport::Any
    # `nothing` if the draw samples nothing. Per draw because one backend compiles
    # a sampling pipeline around its bindings, and compiles are per draw.
    bindings::Any
    # An indexed draw's index buffer; `count` is then the index count.
    indices::Any
    instances::Int
end

"""
One hardware ray-tracing launch, declared like a dispatch: a pipeline, its
arguments and a ray count. The arguments live at a fixed offset in the plan's
`ArgMemory`, written once at `record!`, so a tracing plan can be recorded; packing
them at run time from per-run scratch would leave a recorded plan pointing at
memory the queue has reused.
"""
struct Trace
    pipeline::Any
    accel::Any
    args::Tuple
    # A count, in the same two shapes `Dispatch` takes it: a plain number, or a
    # `DeviceRange` over a count only the GPU knows.
    ndrange::Any
end

"""
    Loop

One [`repeat!`](@ref) loop. Its body passes are compiled once and run `count`
times: every iteration when the count is the host's, or up to `count` with the
device discarding the rest behind its gate. `index` is a one-element `Int32` device
buffer holding the current iteration, from 1, set before the loop and advanced in
the body.
"""
struct Loop
    count::Int
    index::Any
end

mutable struct Pass
    name::String
    kind::Symbol
    targets::Vector{Any}                # render: the colour attachments. copy: the source.
    loads::Vector{LoadOp}        # render only, one per colour attachment
    depth::Any                          # render only, and only if one was given
    depth_load::Union{Nothing,LoadOp}
    dst::Any                            # copy only
    # Default viewport for draws that name none, `nothing` for the whole target. A
    # copy pass: its source region (x, y, width, height), `nothing` for the whole image.
    viewport::Any
    draws::Vector{DrawCall}
    usages::Vector{Pair{Int,Type}}
    dispatches::Vector{Any}
    # `nothing`, or the `(buffer, index)` whose 32-bit value decides whether this
    # pass's work runs (see [`repeat!`](@ref)). Its barriers run either way, so a
    # skipped iteration keeps the ordering intact. A pass property because the
    # graph is a flat list the scheduler reorders.
    predicate::Any
    # `nothing`, or the [`Loop`](@ref) whose body this pass is in. The scheduler
    # keeps a loop's passes together.
    loop::Any
end

abstract type TransientResource <: Resource end

mutable struct TransientBuffer{T,N} <: TransientResource
    # Shaped because resources are interned by identity: `reshape(t, dims)` as an
    # argument would mint a new id per view and hide the hazard between views.
    dims::NTuple{N,Int}
    first::Int
    last::Int
    # The block, not just the address: a Vulkan buffer barrier scopes to
    # (VkBuffer, offset, size).
    block::Any            # ::BufferBlock once placed
    offset::Int
    # Host stores via `setindex!` not yet landed, as `(range, data)` in order, and the
    # flag a run reads first. `hostwritten!` registers the transient as a `CopyDst`
    # of the update pass, so `Liveness` covers the write and the placer does not
    # alias those bytes with anything live.
    pending::Vector{Tuple{UnitRange{Int},Vector{T}}}
    pendinglock::ReentrantLock
    @atomic dirty::Bool
    # Declared at construction, not on first store: the registration pins the
    # interval to pass 1, so the transient can alias nothing. Registering every
    # transient raised SAM 2's encoder peak from 167 MB to 4.7 GB. `setindex!`
    # refuses undeclared transients.
    hostwritten::Bool
end

# Stores start empty.
TransientBuffer{T,N}(dims::NTuple{N,Int}, first::Int, last::Int, block, offset::Int;
                     hostwritten::Bool = false) where {T,N} =
    TransientBuffer{T,N}(dims, first, last, block, offset,
                         Tuple{UnitRange{Int},Vector{T}}[], ReentrantLock(), false,
                         hostwritten)

"""
A transient render target. The backend image exists from construction because the
placer needs its memory requirements; it costs no memory until bound. The five
driver-object fields are type parameters so per-attachment reads while recording
stay static; a backend defines a concrete alias (`VulkanTransientImage`). The graph
reads only `.size` and `.alignment` from `req`.
"""
mutable struct TransientImage{T,F,U,I,R} <: TransientResource
    width::Int
    height::Int
    format::F
    usage::U
    image::I
    req::R
    first::Int
    last::Int
    view::Any
    memory::Any
    # What this target takes its size from, or `nothing` for a fixed size. A window
    # resizes under a plan, and a depth buffer must follow it.
    source::Any
end

"""
Arena kind for buffers; images go to `Images`. Sharing one arena would need
`bufferImageGranularity` padding between linear and optimal-tiled resources, and
getting that wrong is aliasing corruption no validation layer reports.
"""
struct Buffers end

struct Images end

"""
    ArenaPart(kind, index)

The `index`th allocation of an arena whose placement exceeds one allocation
(`maxalloc`, 4 GB on RADV). `Place` spills into further parts, each its own arena
shared across plans; part one is the kind itself. Backends see only
[`basekind`](@ref).
"""
struct ArenaPart{K}
    kind::K
    index::Int
end

"""The kind a backend allocates for: an [`ArenaPart`](@ref)'s own, anything else itself."""
basekind(kind) = kind
basekind(p::ArenaPart) = p.kind

"""The arena key of part `index` of `kind`: the kind itself for the first."""
arenapart(kind, index::Int) = index == 1 ? kind : ArenaPart(kind, index)

"""
Bytes a recording reads while it runs: host-writable, device-addressable, and not
moved or handed out while a recording naming them can replay. Argument blocks and
indirect commands live here; their lifetime belongs to the recording's owner.
"""
struct Unified end

"""
Bytes the host reads after the device wrote them: download staging. Not `Unified`,
whose BAR memory the host reads at write-combined speed (about 29 MB/s); a backend
answers `rawalloc` with host-cached memory where the device has it.
"""
struct Readback end

mutable struct Graph{D}
    dev::D
    passes::Vector{Pass}
    surfaces::Vector{WindowSurface}
    transients::Vector{TransientResource}
    transient_by_id::Dict{Int,TransientResource}
    ids::IdTable          # both directions; see `IdTable`
    # Interning for `use(...; range = ...)`, keyed by value: `ids` is an IdDict, so
    # two equal slices would otherwise get two ids and no hazard between them.
    views::Dict{Tuple{Int,UnitRange{Int}},Any}
end

struct PassHandle
    graph::Graph
    pass::Pass
end

"""
The binding a pass gets from an attribute. A `Buffer` and a `GPURef` of one element
type both erase to `Attr{T}`, differing only in the `stride` field, so a scalar and
a per-element attribute share one pipeline.
"""
struct Attr{T,R} <: Resource
    resource::R
    stride::Int32
end

"""A draw count that lives on the device, wrapped so recording dispatches on it."""
struct Commands{R}
    resource::R
end

"""
A slice of a buffer as its own resource, so passes on disjoint slices get distinct
ids and no barrier between them. `range` is in elements and scopes the barrier; the
kernel still receives the whole buffer.
"""
struct BufferRange
    parent::Any
    range::UnitRange{Int}
end

"""
A shaped window onto bytes the parent owns. The kernel receives the window; the pass
is declared as a slice of the parent over the view's elements, so disjoint views get
no barrier. This lets a graph state an exported model's view buffers directly.
`offset` is in elements of `T` from the parent's first element.
"""
struct ResourceView{T,N,P}
    parent::P
    dims::NTuple{N,Int}
    offset::Int
end

struct CompiledDraw{P,S,A,C,V,B,I}
    # Concrete type parameters: an `Any` field makes the per-draw record path a
    # dynamic lookup that allocates. `P` and `S` are the backend's pipeline and shader.
    compiled::P
    shader::S
    args::A
    count::C
    # Parameters for the same reason: each is `Nothing` or one concrete type per draw.
    viewport::V           # `NTuple{4,Float32}`, already resolved against the pass
    bindings::B
    indices::I            # `nothing`, or the device index buffer; `count` is then indices
    instances::Int
    # Where this draw's arguments live in the plan's argument memory, fixed at compile.
    argoff::Int
    argsize::Int
end

"""
One dispatch, compiled when the plan is, so a frame only writes arguments and
records. A per-frame KernelAbstractions launch recompiles whenever the world counter
changes (any method definition), about 20 ms between acquire and present. `kernel`
is the kernel function after `Adapt`, checked at plan build to be a singleton,
since a closure over device arrays would go stale.
"""
struct CompiledDispatch{L,K,A<:Tuple,I,N,R,O,G}
    # Concrete throughout, as in `CompiledDraw`. `L` is the backend's launch plan.
    launch::L
    iter::I                             # IterPlan for `nd0`
    nd0::N                              # the ndrange the iteration plan was built for
    obj::O                              # the KA kernel, for an ndrange that moves
    # The workgroup `obj` needs at the call, or `nothing` when its type carries one
    # (see `callgroup`).
    group::G
    kernel::K
    args::A
    ndrange::R
    tlas::Bool                          # whether the pipeline was built for ray query
    argoff::Int
    argsize::Int
    # The plan indirect command holding this dispatch's workgroup counts, 0 for a
    # host ndrange. Fixed at compile, like `argoff`.
    indirect::Int
end

"""
A [`Trace`](@ref) resolved by the backend, shaped like [`CompiledDispatch`](@ref).
Core reads only `argsize`, to reserve the argument slot. `accel` and `args` are held
as given and resolved to storage once at `record!`; a value that changes between
runs is a [`GPURef`](@ref), packed as its address.
"""
struct CompiledTrace{P,C,A<:Tuple,R}
    compiled::P
    accel::C
    args::A
    ndrange::R
    argoff::Int
    argsize::Int
    indirect::Int            # as `CompiledDispatch.indirect`; 0 for a host ndrange
end

"""
Argument memory owned by the plan: every draw's and dispatch's arguments, laid out
once at compile, then one 256-byte indirect command per device-sized dispatch.
Written once at `record!` and never again, so one copy suffices and no ring is
needed. A value that changes between runs is a [`GPURef`](@ref): its address is
stored here and an `Update` command writes its value. `store` is a
[`Unified`](@ref) region from the device pool; `token` covers the last run, 0
before the first; `indirect[k]` is dispatch `k`'s view.
"""
mutable struct ArgMemory{S}
    store::S
    address::UInt64
    ptr::Ptr{UInt8}
    # A `RefValue`: a plan's `args` field names the UnionAll, so a plain field would
    # go through dynamic get/setproperty and box the UInt64 on every `run!`.
    # 0 = never run (the first timeline signal is 1).
    token::Base.RefValue{UInt64}
    indirect::Vector{Any}       # [k]: dispatch k's indirect command
end

# Use these, not `am.token` on a `Union{Nothing,ArgMemory}`: a method specialises
# on `ArgMemory{S}` and avoids the boxing.
"""The token covering `am`'s last run; 0 if it never ran or has no memory."""
argtoken(am::ArgMemory) = am.token[]
argtoken(::Nothing) = UInt64(0)

"""Record the token covering `am`'s next submission."""
setargtoken!(am::ArgMemory, v::UInt64) = (am.token[] = v; nothing)

"""
One pass, its compiled draws and dispatches, and the barriers derived to run before
it, including the swapchain image's transition from whatever state acquire left.
"""
struct PassPlan{D,B}
    pass::Pass
    draws::Vector{CompiledDraw}
    # What a compiled dispatch is on this backend: a `CompiledDispatch` when
    # recording commands, a callable when driving KernelAbstractions. A parameter so
    # the per-frame loop specialises.
    dispatches::Vector{D}
    images::Vector{B}        # layout changes, one barrier each
    pre::Vector{Transition}      # what this pass needs before it runs
    barrier::Any                        # one memory barrier covering the rest
    # Whether any dispatch sizes itself on the device; fixed at compile because it
    # decides how the pass is recorded.
    indirect::Bool
end

"""
Two timestamps per pass and a ring of samples. Per plan and per pass, so render
passes, updates and repeated kernels are timed apart. Slots are `2i-1, 2i` for pass
`i` in scheduled order. `pending`: a frame's timestamps are written and unread;
reads skip a frame still in flight instead of waiting. `sampled`: per pass, the
start timestamp of the last sampled frame, so one finished frame is counted once.
"""
mutable struct Profiler{Q}
    pool::Q
    period_ns::Float64
    nslots::Int
    names::Vector{String}
    host_ns::Vector{Vector{Float64}}    # one ring per pass
    gpu_ns::Vector{Vector{Float64}}
    sampled::Vector{UInt64}
    pending::Bool
end

"""The emitter of a backend without recordings: every primitive of the pass walk
does its work on the spot. See `graph/kalaunch.jl`."""
struct Immediate end

mutable struct Plan{D,H<:Tuple}
    graph::Graph{D}
    transitions::Vector{Transition}
    passes::Vector{PassPlan}
    pipelines::Set{Any}
    slabs::Vector{Any}                  # shared region of each arena, not owned
    arenas::Vector{Any}                 # arena key of each region, for remap!/free!
    offsets::Vector{Int}                # per transient, relative to its region
    parts::Vector{Any}                  # per transient, the arena key it was placed in
    peak::Int
    naive::Int
    profiler::Union{Nothing,Profiler}
    alias::Bool                         # kept, so a recompile means what the first one did
    policy::Policy
    # `nothing` when the backend passes arguments directly (see `makeargmemory`).
    args::Union{Nothing,ArgMemory}      # laid out at compile, written at record!
    # The plan's commands, or `nothing` before `record!`. One recording, not one per
    # argument slot, because nothing rewrites argument bytes (see [`ArgMemory`](@ref)).
    recording::Any
    # GPU seconds one submission may take ([`submissionbudget`](@ref)); core
    # partitions the recording from it and `passcost`.
    budget::Float64
    # Whether the caller asked for timings. Otherwise only the measuring recording
    # carries timestamps (see `passcost`).
    profile::Bool
    # Measured GPU seconds per pass, in `passes` order, `NaN` until measured. An
    # unmeasured pass is recorded as its own submission.
    passcost::Vector{Float64}
    # The current recording's pass ranges, so a measurement that changes the
    # partition is noticed.
    partition::Vector{UnitRange{Int}}
    # The loops, as the contiguous range of `passes` each body was scheduled into
    # and the loop it runs. Ascending and disjoint; empty for a plan without one.
    loops::Vector{Tuple{UnitRange{Int},Loop}}
    # Whether the current recording (or walk) writes the profiler's timestamps,
    # so a run only asks for them when there are some to read.
    stamped::Bool
    # Recorded device address => (region, offset) of its eight bytes, baked at
    # `record!`. The region is the `ArgMemory` store or a recording-owned scratch
    # region, whatever `emitinline!` accepts, hence `Any`. When a resource moves
    # (`resize!`, arena growth) the move appends to `pending_patches` and the next
    # `run!` writes the new addresses as commands (see `notify_move!`). Neither is
    # walked per run.
    patchtab::Dict{UInt64,Vector{Tuple{Any,Int}}}
    pending_patches::Vector{Tuple{Any,Int,UInt64}}
    # Every `Buffer` and `GPURef` a pass declares, as a typed tuple: what a host
    # store may be pending on. `run!` lands dirty ones as commands before the
    # recording. Built once by `hostwritten!`, which registers each as a `CopyDst` of
    # the update pass.
    hostwritten::H
    # The recording was discarded by something outside this plan (an images arena
    # grew, a refit); `run!` records again before submitting. Only set when recorded.
    stale::Bool
    # The plan's arena moved (`remap!` rebound its transients). Unlike `stale` this
    # needs a recompile, since `passes` hold concrete device arrays; `remap!` runs
    # mid-growth, so `run!`'s per-frame `refit!` does it.
    replaced::Bool
    # Set at teardown so `free!` and the finalizer cannot both run it. Last because
    # the constructor is positional.
    @atomic freed::Bool
end

"""
The core compilation context. Each phase output is written by one phase and read by
later ones, so a phase can run alone.
"""
mutable struct Compile{D} <: Compilation
    graph::Graph{D}
    alias::Bool
    policy::Policy
    analysis::Analysis
    transitions::Vector{Transition}            # Barriers
    prepass::IdDict{Pass,Vector{Transition}}
    passes::Vector{PassPlan}                          # Pipelines
    pipelines::Set{Any}
end

"""
The Vulkan buffer arena's block: one raw `vkAllocateMemory` with a `VkBuffer` bound
into it, and the buffer's device address. Not a `LavaArray`, whose pool and
finalizing `DataRef` would sit under Mantle's allocator; the Pool owns the Block
and `trim!` frees it.
"""
struct BufferBlock
    buffer::Any
    memory::Any
    address::UInt64
    bytes::Int
    # A `DataRef` whose releaser does nothing, so a transient's `storage` can be a
    # `LavaArray` view for host-side operations while the free stays Mantle's.
    ref::Any
end
