# How the internals in `recording-plan.md` work, as Julia-flavoured pseudocode.
#
# Not runnable. It shows which function owns which decision, what it reads and
# what it writes, and which lock protects what. Helper functions that are not
# the point are left undefined. Native qualification gates remain in the ledger;
# an omitted helper is not an undecided public API.
#
# Normative failure/retirement rules and proof arguments: execution-contracts.md.
# This pseudocode omits helper definitions, not those obligations.
# Sections follow the life of a graph: types, declaration, selection,
# compile, record, run, changes, ordering, memory, holds, eager calls,
# device-type verbs, devices, vendor views.
#
# Locks, in the only order they are taken (plan, "Ordering across graphs and
# queues"): graph lock → the run's hold on its arena parts by id (section 6)
# → window lock → plans' sync locks by plan id (section 8) → sync states by
# id → channel lock.
# Leaves, nothing else taken inside them and no GPU wait inside them: the pool
# lock, the inbox (lock-free), the texture and sampler tables' locks, a plan's
# runend lock (endbetween!, taken under record locks). A GPURef has no
# lock of its own: it is a slot with a sync state, like any shared resource
# (section 7).
# While the host waits (for a shared arena part, a host node's inputs, the
# swapchain acquire, a pressure step's tokens) no sync-state, channel, pool or
# window lock is held; the graph lock and the run's arena-part holds may be
# (section 6). The host never waits for a graph's previous run on the GPU: a
# run's first segment waits for it there, and a GPURef store is
# scheduled. A change waiting for a run to end (stalewait,
# resizing-and-raytracing.jl B.3) holds no lock at all and is never inside a
# run. No Mantle lock is held while vendor code runs (section 14).

# ══ 1. Types ══════════════════════════════════════════════════════════════════

abstract type Device <: KA.Backend end      # LavaDevice, MetalDevice, CudaDevice, ROCmDevice
abstract type Resource end                  # what a graph names and holds: an array's state, an Image, a TLAS, a BLAS, a GPURef
abstract type CompiledCommand end           # CompiledDispatch, CompiledTrace, CompiledCopy, ...
struct HoldDrop{H}                          # a hold to drop once a submission's token is known
    holder::H
end
struct HardwareKey                          # one hardware device under one driver
    type::Type{<:Device}
    uuid::UInt128                           # the driver's device UUID
end

mutable struct Graph
    const device::Device            # the real device it compiles for
    const location::GraphDevice     # what its arrays point at
    # Compile, record, run and free of this graph. A semaphore, not a
    # ReentrantLock: holding a ReentrantLock disables finalizers on the thread
    # (Julia 1.12, base/lock.jl:174), and compile allocates, so acquire! must be
    # able to run the GC under it (section 9). Not reentrant: see lockgraph.
    const lock::Base.Semaphore
    @atomic owner::Union{Nothing,Task}
    @atomic hostcaller::Union{Nothing,Task}   # the task running one of its host functions, or nothing (inhostfunction)
    nodes::Vector{Node}             # as declared; compile copies them (section 4)
    named::IdDict{Resource,Int}     # resources it holds, with the number of declarations naming each (section 10)
    unnamed::Vector{Resource}       # resources no node names any more; their holds drop after the plan naming them retires
    declarer::Union{Nothing,Task}   # the task declaring under the graph lock (nested declarations reuse it)
    rewrites::Tuple                 # opted-in rewriters, empty by default
    const profile::Bool             # opt-in sampled timestamps (execution-contracts.md section 10)
    const priority::Int             # relative scheduling hint; no preemption guarantee
    plan::Union{Nothing,Plan}       # compiled state; its next run applies declarations and deletes (section 5), the graph stays
    @atomic recompile::Bool         # set by markstale! (resizing-and-raytracing.jl B.6), a graph array or window
                                    # beyond its placed size, a failed compile, nodes declared or deleted after compile
    @atomic windowchanged::Bool     # set by resize!(window)
    @atomic freed::Bool
end
# No mode state: groups are values and places in the order (Group, below),
# and when! takes one launch or one draw.

# Graph arrays point at this, never at the Graph. GraphDevice is their
# KernelAbstractions backend; the Graph itself is not a backend (section 2).
struct GraphDevice{D<:Device} <: KA.Backend
    graph::WeakRef
end

# What an array and all its views share: ArrayState, with its storage, host
# view, cell, sync state, holds, freed mark, dependents and readers, is
# defined in resizing-and-raytracing.jl B.2. The finalizer is on that object
# (section 10). Here it also has `source`: the host array a read-only array
# keeps (MantleArray(dev, host; readonly = true)), or nothing.

# One array type for every location L (a device type or GraphDevice).
# Read-only, unified and resizable are runtime
# properties of the resource (ArrayState), not type parameters.
struct MantleArray{T,N,L} <: AbstractGPUArray{T,N}
    location::L
    state::ArrayState
    view::Union{Nothing,ArrayView{N}}    # nothing for a root; a view's static offset, dims and strides
end
struct ArrayView{N}
    offset::Int                         # in elements of the root state
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}              # in elements; nested views compose against the root
end
# A view, reshape or transpose (GPUArrays' `derive`) is the same type and
# shares its parent's state: a store through a view is ordered like one into
# the parent, and a freed parent makes its views unusable. A view has no hold
# of its own; free! and resize! on a view throw (a check of `view`). Its slot
# entries carry its static offset, dims and strides. Conjugating adjoints
# retain their conjugation wrapper; strides alone do not conjugate values.
# Shrinking never unmaps an
# array's pages, so a view never reads unmapped memory
# (resizing-and-raytracing.jl B.2).
Base.size(a::MantleArray) = a.view === nothing ? size(a.state) : a.view.dims
Base.size(st::ArrayState) = size(st, st.lengthsource)
Base.size(st::ArrayState, ::HostLength) = st.dims       # the requested dims (a pending resize! included)
# A length a kernel writes follows Julia's array API:
# size and length wait for the kernel that writes it and read the cell back
# (section 11). During the declaration of a graph that writes it no run has
# produced it, so the call throws, as host reads of contents do.
function Base.size(st::ArrayState, ::DeviceLength)
    declaringwriter(st) &&      # the calling task declares into a graph with a node writing lengthof(a)
        throw(ArgumentError("the length of an array a kernel writes, read while declaring the graph that writes it"))
    return readlength(st)
end

# The array's address and size, where the GPU reads them. Written by applied
# changes (a cell write after resize!, an address after a move;
# resizing-and-raytracing.jl B.3-B.5) or by a kernel through lengthof(a).
# Every plan copies the cell into its argument slots in its head nodes.
struct Cell
    region::Region                  # in the device's cell arena: address::UInt64, dims::NTuple{N,UInt64}
end
struct NoCell end

abstract type Node end

# What compile decides about a node. It lives in the node, and the nodes are the
# plan's own copies (section 4), so a recompile never touches the nodes an older
# plan still in flight uses.
mutable struct CompiledNode
    queue::Union{QueueKind,HostSide}   # schedule!: a queue kind (the run picks the queue), or HostSide() for a host node
    preds::Vector{Node}             # dependencies!: direct predecessors only
    barrier::Barrier                # barriers!, image transitions included
    command::CompiledCommand        # compile!: CompiledDispatch / CompiledTrace / CompiledCopy / …
    cost::Float64                   # costs!
    slots::UnitRange{Int}           # bindings!: its slice of the plan's argument block
end

# Every declared node keeps its origin: the Julia call that launched it (for rewriters).
struct Origin{F,A<:Tuple}
    f::F; args::A
end

mutable struct KernelNode{K,A,R,W,O} <: Node  # dispatch!(location, kernel, args, ndrange)
    const kernel::K; const args::A; const ndrange::R
    const group::Union{Nothing,Int,Tuple}   # requested workgroup dimensions, retained through compile
    const when::W                           # nothing, or a when! flag: indirect count / capacity predication (section 2)
    const origin::O                         # an Origin
    const split::Bool                       # dispatch!(…; split = true): costs! may cut it into workgroup ranges
    compiled::Union{Nothing,CompiledNode}
end
mutable struct TraceNode{P,T,A,R,W,O} <: Node  # trace!(g, pipeline, accel, args, ndrange)
    const pipeline::P; const accel::T; const args::A; const ndrange::R; const when::W
    const origin::O                         # an Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct LoopNode{C,O} <: Node        # a repeated group, made by compile (section 4): launches only
    const count::Int
    const while_nonzero::C                  # nothing, or an integer count on the device (a queue's live size)
    const body::Vector{Node}                # the group's nodes, copied: KernelNodes, TraceNodes, PrepareNodes
    const origin::O                         # an Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct HostNode{F,A,O} <: Node      # HostCall(f); writes declared
    const f::F; const args::A; const writes::Tuple
    const origin::O                         # an Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct ClearNode{T,V,O} <: Node     # clear!(g or group, target, v): a load operation (section 5)
    const target::T; const value::V
    const origin::O                         # an Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct DrawNode{T,P,A,C,I,X,W,O} <: Node   # draw!/when!(g or group, target, pipeline, args, count; …)
    const target::T                         # a Window, an Image, or a tuple of them (colour, then depth)
    const pipeline::P; const args::A; const count::C; const instances::I; const indices::X
    const when::W                           # nothing, or a when! flag: indirect, count × flag (section 2)
    const origin::O                         # an Origin
    compiled::Union{Nothing,CompiledNode}
end

# A group: a value, not a do-block, and a place in its parent's order (the
# graph, or another group), fixed when the group is made. Work is declared
# into it by passing it (dispatch!(group, …), draw!(group, target, …),
# Group(group, name)) anywhere in the code, and runs at the group's place
# whenever it was declared. Not an owner: arrays and allocations belong to
# the device or the graph. It holds GPU work only: no HostCall, build or
# image copy has a method for Group, so those are a MethodError. Its uses
# (section 2): repeat!(group, n) loops it, delete!(g, group) removes it,
# sort!(group) reorders the groups in it; a group of draws is recorded on its
# own, so adding or deleting one recompiles nothing (section 5).
struct Loop{C}
    count::Int
    while_nonzero::C                        # nothing, or an integer count on the device (a queue's live size)
end
mutable struct Group <: Node
    const graph::Graph
    const parent::Union{Graph,Group}
    const name::String
    const nodes::Vector{Node}               # in order: KernelNodes, TraceNodes, ClearNodes, DrawNodes, Groups
    loop::Union{Nothing,Loop}               # set by repeat!
    live::Bool                              # false once deleted: a second delete! throws
end
# Added by Mantle, not declared by the user:
mutable struct HeadNode <: Node             # the first node of every stage of a run (section 5)
    after::Union{Nothing,HostNode}          # the host node it follows; nothing for the run's first
    compiled::Union{Nothing,CompiledNode}
end
# After a launch that writes a's length on the device (LengthPrepare), or
# after a kernel that writes a when! flag (FlagPrepare): writes the launch
# commands that depend on it.
mutable struct PrepareNode{A,K} <: Node
    const array::A
    const kind::K
    compiled::Union{Nothing,CompiledNode}
end
# CopyNode (images, host transfers), RenderNode (a pass compile forms from
# clears and draws, section 5), BuildNode
# (a TLAS or BLAS build or refit), ForeignNode (a closed library), RegionNode
# (a rewriter's merged run), VideoDecodeNode and PresentNode have the same
# shape: data, origin, `compiled`.

# What a plan holds of one array or GPURef in its argument block: the address,
# the size, the KA ndrange values and launch commands over it; for a value
# GPURef, its slot's address, baked by bindings! (the head writes no values:
# a launch reads the value through the slot, a shared resource). A resource
# of the plan like any other, so the head and prepare nodes
# that write it and the launches that read it get edges and barriers from the
# same rules as data.
struct Bound{R}
    resource::R
end

# The host side of a shared resource (a device array, an image, an
# acceleration structure, a GPURef's slot, an arena
# part). Runs are not
# recorded here: what a run did is its segments' tokens, found through the
# plans that name the resource (section 8). Recorded here is only what is not
# a run: eager calls, host reads, vendor views, eviction copies and the
# changes a submission applied ("eager accesses"), plus the pending changes.
struct Access
    token::Token                    # channel and value (section 8)
    stages::Stages
end
mutable struct SyncState
    const id::Int                   # lock order: by ascending id, after the plans' sync locks (section 8)
    const lock::ReentrantLock       # never held while acquire! runs (section 9)
    lastwrite::Union{Nothing,Access}   # eager accesses: the last write,
    reads::Vector{Access}           #   and the reads since, the latest per channel
    retired::Vector{Token}          # the last tokens of plans that named it and were retired (unregister!)
    namers::NamerSet                # the plans naming it, with their use from compile: an immutable snapshot,
                                    #   replaced under this lock by register!/unregister! (section 8);
                                    #   for an arena part, its tenants (all writers)
    pending::Vector{Change}         # changes not yet submitted, in call order (section 8)
    deferred::Vector{HoldDrop}      # hold drops due once the submission applying `pending` is made
                                    #   (resizing-and-raytracing.jl B.4); not changes, so coalescing never drops them
    lent::Bool                      # inside a withview block (section 14)
    transfer::Union{Nothing,HostCopyJob} # internal pageable-copy transaction, distinct from a foreign lease
    freed::Bool                     # an array's creator hold was dropped (section 10); the only `freed` of an array
end

mutable struct Region
    block::Block; offset::Int; bytes::Int
end

# ══ 2. Declaration: arrays on a graph declare, arrays on a device run now ═════

# Where a call runs: the one graph among the arguments if any lives on one (its
# device arrays are then named by that graph), else the arguments' device.
# Two graphs, or two devices, is an error.
location(args...) = foldl(join, (location(a) for a in resources(args)))
location(a::MantleArray) = a.location
location(r::GPURef) = r.device      # a GPURef lives on a device
join(a::Device, b::Device) = a === b ? a : throw(ArgumentError("arguments from two devices"))
join(g::GraphDevice, d::Device) = device(g) === d ? g : throw(ArgumentError("arguments from two devices"))
join(d::Device, g::GraphDevice) = join(g, d)
join(a::GraphDevice, b::GraphDevice) = a === b ? a : throw(ArgumentError("arguments from two graphs"))

# GraphDevice is a KernelAbstractions backend: KA launches, broadcasting,
# GPUArrays' generic code and KA libraries run on graph arrays and declare
# into the graph. Only host reads
# and writes of contents throw, at their line: during declaration no run has
# produced them, and data the host supplies per run is a device array (section 7).
Base.getindex(::MantleArray{T,N,<:GraphDevice}, i...) where {T,N} = nodata()
Base.setindex!(::MantleArray{T,N,<:GraphDevice}, v, i...) where {T,N} = nodata()
KA.get_backend(a::MantleArray) = a.location         # the array's device, or its graph's GraphDevice
KA.allocate(l::GraphDevice, ::Type{T}, dims::Tuple) where {T} = MantleArray(l, T, dims...)   # a transient of the graph
# A KA launch over graph arrays (a kernel object built for a GraphDevice) adds
# a KernelNode. Its ndrange is what the traced code computed on the host: the
# node records it together with the requested dims of the argument arrays at
# that point, and a resize of one of them makes the next compile throw, since
# the host code that computed the range does not run again (checkrange,
# section 4). What the host computes from its own values cannot be detected
# and becomes a constant of the plan; values that change per run are GPURefs.
(k::KA.Kernel{<:GraphDevice})(args...; ndrange, workgroupsize = nothing) =
    dispatch!(KA.backend(k), k.f, args, HostRange(ndrange, hostdims(args));
              group = launchgroupsize(k, workgroupsize), origin = Origin(k, args))
struct HostRange{R}
    ndrange::R
    dims::Vector{Pair{ArrayState,NTuple}}   # each argument array's requested dims when the range was computed
end

# The one launch verb. On a graph device it adds a node; on a group, a launch
# of that group; on a device it runs now (section 11). An operation's method
# calls it and does not know which.
dispatch!(l::Union{GraphDevice,Group}, kernel, args::Tuple, ndrange;
          group = nothing, origin = Origin(kernel, args), split = false) =
    declare!(l, KernelNode(kernel, args, ndrange, group, nothing, origin, split, nothing))
# The KA adapter
# resolves the size fixed in k's type when no launch-time override is given;
# compile consults config(dev, f) only if neither supplied a size.
dispatch!(g::Graph, args...; kw...) = dispatch!(g.location, args...; kw...)
function declare!(l::Union{GraphDevice,Group}, p::KernelNode)
    declaring(graph(l)) do
        join(graph(l).location, location(p.args...))   # a group's arguments live on its graph or its device
        foreach(usable, resources((p.when, p.args...)))   # first: a throw leaves the graph unchanged
        checkwritable(p.kernel, p.args)
        addnode!(l, p)
        for a in lengthwrites(p.args)               # arguments of the form lengthof(a); a is resizable
            addnode!(l, PrepareNode(a, LengthPrepare(), nothing))   # writes the launch commands and sizes over a
        end
    end
    return nothing
end
graph(gr::Group) = gr.graph

# A write touch on a read-only array throws at the call, declared or eager:
# from the per-argument access dispatch! derives at declaration (api.md, section 3),
# the same access uses(p) reads at compile. No compile is involved.
function checkwritable(kernel, args)
    for (r, u) in argumentuses(kernel, args)        # per argument: the resource, and Read() or Write()
        writes(u) && isreadonly(r) &&
            throw(ArgumentError("a launch writes a read-only array (readonly = true)"))
    end
end
# A runtime property of the resource: the host array that
# MantleArray(dev, host; readonly = true) keeps, or nothing.
Base.isreadonly(st::ArrayState) = st.source !== nothing
Base.isreadonly(::Union{GPURef,AccelStructure,Image}) = false    # no read-only form

# A graph is never replaced for a structure change. Nodes declared into a
# compiled graph, or a group deleted from it, are the next run's to apply
# (applychanges!, section 5): groups of draws are recorded on their own, any
# other change compiles a new plan from the graph's nodes and retires the old
# one after its last run (recompile!). The graph, its windows, the resources
# it names and the device arrays it reads stay. Work that will be removed is
# declared into a group:
#   plot = Group(g, "plot")              # its place: the end of g's order, now
#   draw!(plot, target, pipeline, args, n)
#   delete!(g, plot)                     # the plot removed
# Nodes declared into the graph itself stay until free!(g). A graph array's
# contents live only during a run, so what must survive runs and recompiles
# (a film, an accumulation count) is a device array the graph names.
# Declaring takes the graph lock, so it waits while a run of g has stages
# left; a declaration inside another declaration of the same task runs under
# the lock already held. A host function of g that declares into g throws
# (lockgraph).
declarable(g::Graph) = (@atomic g.freed) && throw(ArgumentError("declaring into a freed graph"))
declaring(f, g::Graph) = g.declarer === current_task() ? f() : lockgraph(g) do
    declarable(g)
    g.declarer = current_task()
    try f() finally g.declarer = nothing end
end
graph(g::Graph) = g
graph(gr::Group) = gr.graph
nodelist(g::Graph) = g.nodes
nodelist(gr::Group) = gr.nodes
islive(::Graph) = true
islive(gr::Group) = gr.live && islive(gr.parent)
isgroup(::Group) = true
isgroup(::Node) = false
# A node joins the end of its parent's order and names its resources. With a
# plan, the change is recorded for the next run, which decides whether it is
# a piece change or a recompile (applychanges!, section 5).
function addnode!(parent::Union{Graph,Group}, n::Node)
    islive(parent) || throw(ArgumentError("declaring into a deleted group"))
    checkloop(parent, n)
    push!(nodelist(parent), n)
    g = graph(parent)
    name!(g, n)
    g.plan === nothing || push!(g.plan.changes, Added(parent, n))
end
addnode!(l::GraphDevice, n::Node) = addnode!(graph(l), n)
# A repeated group holds launches only (repeat!, below).
checkloop(::Graph, n) = nothing
checkloop(gr::Group, n) = gr.loop === nothing || loopable(n) ||
    throw(ArgumentError("a draw, a clear or a group declared into the repeated group $(gr.name)"))
loopable(::Union{KernelNode,TraceNode}) = true
loopable(::Node) = false
# One node from a declaring call (build!, refit!, a clear, a draw, a copy).
declare!(l::Union{Graph,Group}, n::Node) = declaring(() -> (foreach(usable, heldresources(n)); addnode!(l, n)), graph(l))

function Group(parent::Union{Graph,Group}, name)
    declaring(graph(parent)) do
        gr = Group(graph(parent), parent, String(name), Node[], nothing, true)
        addnode!(parent, gr)                        # its place: the end of the parent's order, now
        return gr
    end
end
# The group and everything in it leave the order; their resources lose
# their names. With a plan, the next run decides: a group of draws leaves its
# pass's list, any other group recompiles (section 5).
Base.delete!(g::Graph, gr::Group) = declaring(g) do
    gr.graph === g && islive(gr) || throw(ArgumentError("delete! of a group not in this graph (deleted already?)"))
    gr.live = false
    filter!(n -> n !== gr, nodelist(gr.parent))
    g.plan === nothing ? unname!(g, declared(gr)) : push!(g.plan.changes, Deleted(gr))
end
# The groups a group holds, reordered (by sees the groups); its other nodes
# keep their places. The next run rewrites the pass's list where they are
# groups of draws in one pass, and recompiles otherwise (section 5).
function Base.sort!(gr::Group; by = identity, lt = isless, rev = false)
    declaring(gr.graph) do
        slots = findall(isgroup, gr.nodes)
        gr.nodes[slots] = sort(gr.nodes[slots]; by, lt, rev)
        gr.graph.plan === nothing || push!(gr.graph.plan.changes, Reordered(gr))
        return gr
    end
end
# A resource no remaining node names leaves `named`; while a plan names it,
# its hold drops only after that plan's last run (recompile!; release!).
declared(n::Node) = (n,)
declared(gr::Group) = [gr; [x for n in gr.nodes for x in declared(n)]]   # its loop count, then its nodes
function unname!(g::Graph, nodes)
    for n in nodes, st in unique(heldresources(n))   # a Group: its while_nonzero count's
        (g.named[st] -= 1) == 0 || continue
        delete!(g.named, st)
        g.plan === nothing ? drophold!(st) : push!(g.unnamed, st)
    end
end

# Operations are methods on MantleArray. Each launches its kernels on
# location(...) over launchrange(...); selection is a call here, at declaration.
function Base.fill!(a::MantleArray, v)
    dispatch!(location(a), fill_kernel!, (a, v), launchrange(a); origin = Origin(fill!, (a, v)))
    return a
end

# launchrange(a) = LengthOf(a): a reference to the array, resolved at compile
# (direct while the array is fixed, indirect after its first resize!);
# resizing-and-raytracing.jl B.7.

function LinearAlgebra.mul!(C::MantleArray, A::MantleArray, B::MantleArray)
    l = location(C, A, B)
    k = select(device(l), candidatesfor(l), mul!, C, A, B)   # the device's table (section 3)
    launch!(l, k, C, A, B; origin = Origin(mul!, (C, A, B)))
    return C
end
# launch!(l, k::KernelCandidate, …) is dispatch!; launch!(l, k::ForeignCandidate, …)
# adds a ForeignNode on a graph and calls the library now on a device.

# Two kernels with a temporary between them, sized from `a` by a size
# expression evaluated at compile (SizedBy, resizing-and-raytracing.jl B.7).
# On a device the temporary is a region retired with its last use; on a graph
# it is a transient that liveness! and place! handle like any other.
function GPUArrays.mapreducedim!(f, op, r::MantleArray, a::MantleArray; init = nothing)
    l = location(r, a)
    MantleArray(l, eltype(r), SizedBy(nblocks, a)) do partial
        dispatch!(l, partials_kernel!, (f, op, partial, a, init), launchrange(a))
        dispatch!(l, finish_kernel!, (op, r, partial, a), launchrange(r))
    end
    return r
end

# Broadcast: the Broadcasted tree is data; one kernel node.
function Base.copyto!(dest::MantleArray, bc::Broadcast.Broadcasted)
    l = location(dest, arrays(bc)...)
    dispatch!(l, broadcast_kernel!, (dest, bc), launchrange(dest); origin = Origin(broadcast!, (bc.f, dest, bc.args...)))
    return dest
end

# copyto! between arrays in a graph is a copy kernel, so it may be declared into
# a Group (a launch, not a copy node). Mantle operations also have an explicit
# leading-Group method; it passes the group through every internal launch and
# filters selection to kernels. Allocations still use the graph's location.
Base.copyto!(dst::MantleArray, src::MantleArray) =
    copyto!(location(dst, src), dst, src)
Base.copyto!(l::Union{Device,GraphDevice,Group}, dst::MantleArray, src::MantleArray) =
    (dispatch!(l, copy_kernel!, (dst, src), launchrange(src)); dst)

# repeat!(group, n; while_nonzero): the group runs n times per run at its
# place, or until while_nonzero, an integer count on the device (a queue's
# live size), is 0. No do-block body: a body would invite host code between
# dispatches. A repeated group holds launches only (kernels, traces, copy
# kernels): a draw, a clear or a group in it throws, here or when declared
# later (checkloop), because a render pass would need its targets' layout to
# be the same on the way in and out of every iteration, and loops do not
# nest. A trace in it is indirect, and the loop head zeroes its count once
# the loop ends. Before each iteration a loop-head kernel writes the index
# into the slots of the launches that take iteration(group).
# Compile makes the group's LoopNode (section 4).
iteration(gr::Group) = LoopIndex(gr)   # a placeholder argument; the loop head writes i into its slots
function repeat!(gr::Group, count::Int; while_nonzero = nothing)
    declaring(gr.graph) do
        gr.loop === nothing || throw(ArgumentError("repeat! of group $(gr.name), repeated already"))
        all(loopable, gr.nodes) || throw(ArgumentError("repeat! of group $(gr.name): it holds a draw, a clear or a group"))
        while_nonzero === nothing || usable(while_nonzero)
        gr.loop = Loop(count, while_nonzero)
        name!(gr.graph, gr)                         # the count (heldresources of a Group)
        gr.graph.plan === nothing || (@atomic gr.graph.recompile = true)
    end
    return gr
end
# The form for a loop of one launch: a group of one.
function repeat!(g::Graph, kernel, args::Tuple, ndrange; max::Int, while_nonzero = nothing)
    gr = Group(g, string(nameof(kernel)))
    dispatch!(gr, kernel, args, ndrange)
    return repeat!(gr, max; while_nonzero)
end

# when!: ONE launch that runs only where `flag` is true
# on the GPU. The flag is a Bool by the methods' argument types: a
# GPURef{Bool} or a one-element Bool array; any
# other type is a MethodError by dispatch, no runtime check. The kernel is a
# function: a HostCall, an operation or a build is a MethodError too. Several
# launches under one flag are several when!s. trace! has the same methods with a
# TraceNode; when!(g or group, target, flag, …) is one draw (section 5).
when!(g::Graph, flag::Union{GPURef{Bool},MantleArray{Bool,1}}, kernel::Function, args::Tuple, ndrange; kw...) =
    when!(g.location, flag, kernel, args, ndrange; kw...)
when!(l::Union{GraphDevice,Group}, flag::Union{GPURef{Bool},MantleArray{Bool,1}}, kernel::Function, args::Tuple, ndrange;
      origin = Origin(kernel, args)) =
    declare!(l, KernelNode(kernel, args, ndrange, nothing, flag, origin, false, nothing))   # named through the node, like its arguments
# A draw's visibility is a store into its flag, never a new plan. The
# mechanism repeat!'s loop head uses: on Vulkan/Metal a when! launch/draw
# uses an indirect command/range whose count is multiplied by the flag
# (values other than 0 and 1 cannot exist). That is the head for a flag the
# host writes (a store or a GPURef), and a PrepareNode(flag,
# FlagPrepare()) right after the kernel that writes it otherwise (compile
# inserts it, section 4). No conditional rendering: the flag is read by
# address like any argument, so it may move and is never a handle dependence,
# and Metal, which has no conditional rendering, needs nothing else. The
# meaning is the same on every device type. CUDA/HIP use the fixed-capacity
# recorded launch with entry predication as the portable baseline; optional
# native conditional nodes require separate qualification (execution-contracts.md
# section 3). Core expands bounded repeated groups before DAG analysis.

# A host function inside a call is a node; Mantle schedules it (section 6).
# No method for a Group, and not a when! launch.
function dispatch!(l::GraphDevice, h::HostCall, args::Tuple; writes::Tuple)
    g = graph(l)
    declaring(g) do
        foreach(usable, resources(args))
        addnode!(g, HostNode(h.f, args, writes, Origin(h.f, args), nothing))
    end
end

# ══ 3. Selection: a call in the operation's method ════════════════════════════

# The device's measured table, else core's estimate over the candidates the
# device type contributes. Called at declaration: the graph's device is known.
# In a Group only kernel candidates qualify: a foreign node (cuBLAS,
# MPSGraph) cannot run under a device-decided condition. (when! takes a
# kernel, not an operation.)
candidatesfor(::Union{Device,GraphDevice}) = AnyCandidate()
candidatesfor(::Group) = KernelsOnly()
function select(dev::Device, allowed, f, args...)
    key = SelectionKey(f, args...; config = config(dev, f))   # types, shapes/capacities, layout, alignment, implementation version
    eligible = k -> supports(k, key) && satisfies(caps(dev), requires(k)) && qualifies(k, allowed)
    measured = lock(selectionlock(dev)) do
        get(selectiontable(dev), (key, allowed), nothing)      # immutable, versioned entry
    end
    measured === nothing || !eligible(measured) || return measured
    cands = filter(eligible, allcandidates(dev, f))
    isempty(cands) && throw(ArgumentError("no eligible implementation for this operation"))
    return argmin(k -> estimate(dev, k, key), cands)
end
# allcandidates combines core's portable candidates with candidates(dev, f).
# Measured entries are validated against the same requirements and compiler
# configuration before returning them. Tuning is explicit, outside run! and
# outside the graph being declared; it uses private inputs and output/workspace.
qualifies(k, ::AnyCandidate) = true
qualifies(k::KernelCandidate, ::KernelsOnly) = true
qualifies(k, ::KernelsOnly) = false

# ══ 4. Compile ════════════════════════════════════════════════════════════════

# A fixed sequence of steps, each a function of the compile state. Record and
# run may not inspect the graph or the device; this is where that happens.
# The state holds copies of the graph's nodes in order, groups resolved
# (flatten): the plan owns them and their compiled data.
function compile(g::Graph)
    s = CompileState(g, flatten(g.nodes, g))
    lowerloops!(s)                  # core expansion, before dependencies/costs/placement
    pushfirst!(s.nodes, HeadNode(nothing, nothing))
    for h in hostnodes(s.nodes)     # each later stage copies the cells again
        insertafter!(s.nodes, h, HeadNode(h, nothing))
    end
    for (w, flag) in flagwriters(s.nodes)           # a when! flag a kernel of this graph writes:
        insertafter!(s.nodes, w, PrepareNode(flag, FlagPrepare(), nothing))   # its when! launches' commands × flag (section 2)
    end
    layout!(s)                      # the Bound resources, before anything reads them
    formpasses!(s)                  # clears and draws into the same targets: one RenderNode each (section 5)
    isempty(g.rewrites) || (dependencies!(s); foreach(r -> r(s), g.rewrites))
    dependencies!(s)
    foreach(n -> compile!(s, n), s.nodes)
    foreach(f! -> f!(s), (costs!, schedule!, hostcuts!, presentspan!, liveness!, renderstores!, place!, sharing!,
                          barriers!, submissions!, summary!, bindings!))
    return Plan(s)      # stages, placement, argument block, tables; starts marked: unmapped,
end                     #   restore; running = false; compiles = 1 (recompile! carries the count)

# The order with groups resolved: a group's nodes at its place, a repeated
# group as one LoopNode over copies of its nodes. Each copy of a draw keeps
# the group that holds it (its holder), so formpasses! can cut a pass's draws
# into pieces.
flatten(nodes, holder) = Node[x for n in nodes for x in flatten(n, holder)]
flatten(n::Node, holder) = (deepcopynode(n, holder),)
flatten(gr::Group, holder) = flatten(gr, gr.loop)
flatten(gr::Group, ::Nothing) = flatten(gr.nodes, gr)
flatten(gr::Group, l::Loop) = (LoopNode(l.count, l.while_nonzero, flatten(gr.nodes, gr), Origin(repeat!, (gr, l.count)), nothing),)

# LoopNode is compile-time syntax only. The baseline adds ordinary prepare/
# dispatch nodes to the SAME DAG; drivers never walk or choose a loop algorithm.
function lowerloops!(s)
    s.nodes = Node[q for p in s.nodes for q in lowerloop(s, p)]
end
lowerloop(s, p::Node) = (p,)
function lowerloop(s, p::LoopNode)
    p.count == 0 && return ()
    control = loopcontrol!(s, p)   # plan-owned Bound state/slots, no source Cell
    nodes = Node[PrepareNode(control, LoopStartPrepare(), nothing)]
    for i in 1:p.count
        head = PrepareNode(control, LoopHeadPrepare(i, p.while_nonzero), nothing)
        body = iterationcopy(p.body, control, i) # shared resources; distinct argument/command slots
        append!(nodes, (head, body...))
        loopuses!(s, control, head, body) # head writes control; EVERY body node reads it
    end
    return nodes
end
# Start clears the stopped latch once per run. Each head reads the device count,
# latches stop once it is zero and writes its body's iteration/predicate slots.
# Subsequent heads cannot revive a stopped loop. Prepare uses describe control
# slots and count reads; bodies keep their real uses/origins. Loop-control uses
# are included in uses(p,s), so dependencies! derives head→body→next-head
# after rebuilding the DAG, even for a body that ignores iteration(). Recompute hazards,
# lifetimes and costs after expansion. Cutter prefers retained iteration boundaries;
# indivisible over-budget work follows execution-contracts.md §3.

# What each node touches. The Bound resources are the plan's argument slots.
# A GPURef argument is a read of its slot (argumentuses), a shared
# resource whose address the plan bakes.
uses(p::KernelNode, s) = (loopuses(p, s)..., argumentuses((p.kernel, p.args...))..., (Bound(r) => Read() for r in bound((p.kernel, p.args...)))..., flaguses(p.when)...)
uses(p::TraceNode, s) = (loopuses(p, s)..., argumentuses((p.accel,))...,
    (Bound(r) => Read() for r in bound((p.accel,)))...,
    programuses(p.pipeline, p.args, s)..., flaguses(p.when)...) # slots/captures across all stages
uses(p::HeadNode, s) = (                    # everything a head reads and writes
    headafter(p.after)...,                                 # a later stage's head runs after its host node
    (Bound(r) => Write() for r in boundresources(s))...,   # addresses, sizes, launch commands
    (r => CellRead() for r in celled(s))...,               # each device or resizable array's cell
    (f => Read() for f in hostflags(s))...)                # the when! flags the host writes: their counts × flag
headafter(::Nothing) = ()
headafter(h::HostNode) = (HostDone(h) => Read(),)
uses(p::PrepareNode, s) = (loopuses(p, s)..., uses(p, s, p.kind)...)
uses(p::PrepareNode, s, ::LoopStartPrepare) = (p.array => Write(),)
uses(p::PrepareNode, s, k::LoopHeadPrepare) = (p.array => Write(), flaguses(k.count)...)
uses(p::PrepareNode, s, ::LengthPrepare) = (p.array => CellRead(), Bound(p.array) => Write())
uses(p::PrepareNode, s, ::FlagPrepare) = (p.array => Read(), (Bound(q) => Write() for q in conditionedby(s, p.array))...)
uses(p::HostNode, s) = (hostreads(p.args)..., (w => Write() for w in p.writes)..., HostDone(p) => Write())
# Members compile does not know (push! and delete! change them without
# recompiling) get a placeholder. A node that reads through a composite (a
# build or refit of a structure, a trace or ray query of a TLAS, whose hit
# shaders read BLAS geometry) also reads AnyWrite: it conflicts with every
# earlier write in the plan, and every later write conflicts with it.
# dependencies! and barriers! match it against every key, not by name, which
# gives the edges and a full memory barrier. At run the members are known:
# expand (B.4) turns the build into a write of the structure and reads of its
# members, and a read of a composite into reads of its members, for waits
# and records.
uses(p::BuildNode, s) = (p.structure => BuildUse(), AnyWrite() => Read())
argumentuse(t::TLAS, ::Read) = (t => Read(), AnyWrite() => Read())   # traces and ray queries, inside argumentuses
uses(p::RenderNode, s) = (renderuses(p, s)..., AnyWrite() => Read())
# renderuses unions the actual targets and shader accesses of all pieces.
# AnyWrite reserves the ordering needed by future read-only pieces, including
# edges to producers on other queues; it does not perform layout transitions.

# ── dependencies!: an edge "q before p" wherever two nodes conflict ──
# One hazard function, shared with barriers! and the ordering across graphs.
hazard(before::Use, after::Use) =
    !(reads(before) && reads(after)) && !(unordered(before) && unordered(after))

function dependencies!(s)
    last = Dict{Resource,Vector{Tuple{Node,Use}}}()
    for p in s.nodes
        us = mergeduses(p, s)      # union aliases' accesses; a write is not lost to a read
        for (r, u) in us, (q, v) in earlier(last, r)
            hazard(v, u) && addedge!(s.dag, q, p)
        end
        # Publish only after checking every use: alias arguments and AnyWrite
        # must never produce an edge from a node to itself.
        for (r, u) in us
            push!(get!(last, r, []), (p, u))
        end
    end
    # The head writes every Bound and every launch reads its own, so the head
    # precedes every root without a special case. Keep only direct predecessors
    # (what a CUDA/HIP graph node takes), but never drop an edge whose only
    # replacement runs through a conditional node: that node may not run.
    transitivereduce!(s.dag; through = n -> !conditional(n))
end
# The earlier uses a use conflicts with: those of the same key, and an
# earlier read through a composite (AnyWrite, read anything: a later write
# must follow it); for AnyWrite itself, every earlier use of any key, of
# which hazard keeps the conflicting ones.
earlier(last, r) = [get(last, r, ()); get(last, AnyWrite(), ())]
earlier(last, ::AnyWrite) = [e for es in values(last) for e in es]

# ── compile!: one method per node type; no lowering here ──
# CompiledDispatch(kernel, extents, arguments): arguments are InBlock() (the
# plan's argument block at the node's slots) in a graph, the packed region in
# an eager call (section 11). launchextents is Indirect for a when! launch
# (its command's count is multiplied by the flag, section 2).
function compile!(s, p::KernelNode)
    checkrange(p.ndrange)
    p.compiled = CompiledNode()
    p.compiled.command = CompiledDispatch(kernel!(s.device, p.kernel, argtypes(p.args); group = p.group),
                                          launchextents(s.device, p), InBlock())
end
# A KA launch's ndrange came from the host (HostRange, section 2): an
# argument array resized since its declaration makes the compile throw.
checkrange(r::HostRange) = all(((st, dims),) -> st.dims == dims, r.dims) ||
    throw(ArgumentError("a KA launch's ndrange was computed on the host from an array resized since; declare it again"))
checkrange(_) = nothing
function compile!(s, p::TraceNode)
    p.compiled = CompiledNode()
    p.compiled.command = CompiledTrace(pipeline!(compiletrace, s.device, p.pipeline, argtypes(p.args)),
                                       launchextents(s.device, p), InBlock())
end
# A render pass (formed by formpasses!): the compiled render keeps its
# targets and their load and store operations; each piece of p.pieces is
# compiled on its own with compilepiece!, the same call a new group of draws
# gets at run (section 5): its pipelines, argument region and cell-copy
# entries, into the plan's list for the pass. It calls core's
# pipeline!(compiledraw, dev, pipeline, argtypes, formats); native compilation
# happens on a cache miss. Shader code identity and configuration are in the
# key. The cache algorithm belongs to core for every device type (api.md 4.4).
function compile!(s, p::RenderNode)
    p.compiled = CompiledNode()
    p.compiled.command = CompiledRender(p.targets, p.loads, ())   # renderstores! resolves stores after liveness!
    setpieces!(s, p, [compilepiece!(s, piece) for piece in p.pieces])   # recorded with the plan (recordpiece)
end
compile!(s, p::HeadNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledHead(headkernels(s.device)...))
compile!(s, p::PrepareNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledDispatch(kernel!(s.device, prepare_kernel!, preparetypes(p)), 1, InBlock()))
compile!(s, p::HostNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledHost(p.f, p.args))

# The `Kernel` lives at the kernel's CodeInstance: `kernelfor` finds it
# through `GPUCompiler.cached_results` (one per compiler configuration), with the
# access per argument set at declaration. compilekernel fills its code if
# missing and its entry for `dev` (native handle and KernelInfo) under the
# kernel's lock. Idle entries can be dropped/recreated by core's bounded native
# cache (execution-contracts.md section 13). It reports whether this call compiled for the per-device
# compile counter. No other kernel table exists, in core or in a device type.
function kernel!(dev, f, types; group = nothing)
    cfg = withgroup(config(dev, f), group)             # explicit size wins; part of the compiler/cache configuration
    k = kernelfor(dev, f, types, cfg)                  # the Kernel at the CodeInstance; access already set
    native, compiled = compilekernel(dev, k, cfg)
    compiled && increment!(dev.compiles)
    return entry(k, dev)                               # the DeviceKernel: native and KernelInfo
end

# ── costs!: estimated, never measured-then-re-recorded ──
# A launch runs whole. One that cannot finish within the job timeout is a
# compile error, unless its caller opted in to cutting it: then, where it
# exceeds the budget, it becomes workgroup ranges. The estimate is a lower
# bound (the work at full throughput and bandwidth), so the error fires only
# for a launch that cannot finish in time. jobtimeout = Inf checks nothing.
function costs!(s)
    budget, timeout = submissionbudget(caps(s.device)), caps(s.device).jobtimeout
    for p in copy(s.nodes)
        p.compiled.cost = estimate(s.device, p)
        if cuttable(p) && p.compiled.cost > budget
            cutranges!(s, p, budget)
        elseif p.compiled.cost > timeout
            throw(LaunchTooLong(p.origin, p.compiled.cost, timeout))
        end
    end
end
cuttable(p::KernelNode) = p.split               # a library candidate with `partition` is cut into its
cuttable(p) = false                             #   tiles the same way (execution-contracts.md §3)
# The launch becomes consecutive ranges of its workgroups along its last axis,
# each estimated within the budget: one KernelNode per range, with the
# launch's kernel, arguments, uses and origin and the range's `firstgroup`,
# which the entry wrapper adds to the workgroup id, so the kernel sees the
# full grid. The ranges take the launch's place in the order; submissions!
# cuts between them as between any nodes.
function cutranges!(s, p, budget)
    groups = numgroups(p)                           # along the last axis of the ndrange
    n = min(groups, ceil(Int, p.compiled.cost / budget))
    replace!(s, p => [rangenode(p, r) for r in Iterators.partition(1:groups, cld(groups, n))])
end
estimate(dev, p) = estimate(caps(dev), p) * calibration(dev, p)
estimate(c::Caps, p) = max(work(p) / c.throughput[eltype(p)], bytes(p) / c.bandwidth)
    # work: 2MNK for a mul!, else from KernelInfo and the ndrange (the capacity for a kernel-written length)
    # c.throughput, c.bandwidth: facts the device type reports (api.md, `caps` in 4.1)
    # calibration: from past profiled runs of this device, 1.0 without any

# ── schedule!: order and queue kind per node ──
# Compile decides the kind; the queue of that kind is picked at each run
# (pickchannel, section 8). `s.main` is the kind most of the graph runs on:
# Main() for a graph with graphics work, otherwise the compute-capable kind
# with the most queues (Compute() on RADV: one Main queue, four Compute queues).
function schedule!(s)
    s.order = toposort(s.dag)
    s.main = any(hasgraphics, s.nodes) ? Main() : widestcompute(queuekinds(s.device))
    foreach(p -> (p.compiled.queue = queuefor(s, p)), s.order)
end
queuefor(s, ::HostNode) = HostSide()
queuefor(s, ::HeadNode) = s.main
queuefor(s, ::VideoDecodeNode) = VideoDecode()
queuefor(s, p::CopyNode) = queuefor(s, p, independenthostcopy(p), caps(s.device).transfer)
queuefor(s, p::CopyNode, ::Independent, ::HasTransfer) = Transfer()
queuefor(s, p::Node) = overlapgain(p) > caps(s.device).crossqueuecost ? othercompute(s.main) : s.main
othercompute(::Main) = Compute()
othercompute(::Compute) = Main()

# ── hostcuts!: before liveness!, because it changes the order ──
# Moves independent nodes across host nodes, merges adjacent host nodes, and
# inserts copies between device placement and host-visible memory: before the
# host node for what it reads, after it for what it writes.
function hostcuts!(s)
    for h in hostnodes(s.order)
        moveacross!(s, h)           # a node with no path to or from h goes to the side with less work
    end
    mergeadjacent!(s, HostNode)
    for h in hostnodes(s.order), (a, u) in devicearrayuses(h)
        c = insertcopy!(s, h, a, u) # a new node with its edges
        compile!(s, c); c.compiled.cost = estimate(s.device, c); c.compiled.queue = queuefor(s, c)
    end
end

# ── liveness!, place!: placement stored in the plan ──
function liveness!(s)
    for (i, p) in enumerate(s.order), r in transients(p)
        s.first[r] = min(get(s.first, r, i), i)
        s.last[r]  = max(get(s.last, r, i), i)
    end
    # a repeated group's transients live for one iteration unless read before written
end
function renderstores!(s)
    for p in rendernodes(s.order)
        p.compiled.command = CompiledRender(p.targets, p.loads, storeops(s, p))
    end
end
function place!(s)
    items = [Item(r, s.first[r], s.last[r], bytes(capacity(r))) for r in keys(s.first)]
    # Transients a host node reads or writes go to a host-visible arena part,
    # which is not sparse, so a host function gets one contiguous view. Arena
    # parts are shared with other graphs, host nodes or not; a run holds its
    # parts until its last stage is submitted (holdparts!, section 6).
    s.offsets = place(Problem(items), BestFit(); part = r -> parttype(s, r))   # host-visible if a host node uses r
    s.peak = maxload(s.offsets)                     # interval packing, memory/placement.jl
end

# ── sharing!: transients placed on the same bytes conflict ──
# Edges for CUDA/HIP graphs; barriers! finds the same conflicts by bytes.
function sharing!(s)
    for (a, b) in overlapping(s.offsets)            # a's last use before b's first
        addedge!(s.dag, lastuser(s, a), firstuser(s, b))
    end
end

# ── barriers!: by memory, not by name; merged over conditional launches ──
# A transient's key is its (arena part, byte range), so two transients placed on
# the same bytes conflict here although the graph never names them together.
function barriers!(s)
    state = MemoryState()
    for p in s.order
        for (r, u) in uses(p, s), (k, v) in overlapping(state, memkey(s, r))
            hazard(v, u) && add!(p.compiled.barrier, barrier(v, u))
        end
        after = apply(state, p, s)
        state = conditional(p) ? merge(state, after) : after    # later nodes are correct either way
    end
end
# memkey(s, AnyWrite()) overlaps every key: the node
# gets a global memory barrier (VkMemoryBarrier2 without a resource, one per
# build node; device memory caches only, no layout change).

# ── submissions!: stages of segments, cut by budget, queue, host nodes, present ──
# A stage is the segments between two host nodes, and the host node after them.
function submissions!(s)
    c = Cutter(submissionbudget(caps(s.device)))    # core's policy from the device's facts: a smaller target
                                                     #   where caps(dev).displayattached
    foreach(p -> addnode!(s, c, p), s.order)
    finish!(s, c)                                    # closes the last stage, also after a trailing host node
    # Within a run, a segment waits on the segments holding its predecessors on
    # other queues (for CUDA/HIP: predecessors in other segments); these waits
    # are passed to submit! with the others (section 6).
    crossqueuewaits!(s)
end
# A host node closes the stage; any other node joins the open segment unless the
# queue changes, the budget is reached, or it is the first node that writes
# the swapchain image. That node starts the per-image present segment, which
# runs to the PresentNode: the segment waits on the acquire and is recorded per
# image and generation (section 7), so nothing that writes the image is
# recorded once outside it.
addnode!(s, c, p::HostNode) = (c.segment.presents && hostinpresent(p);   # throws (below)
    closesegment!(c); c.stage.host = p; push!(s.stages, c.stage); c.stage = Stage())
# A host node between the first write of the swapchain image and its
# PresentNode that hostcuts! could not move out (it depends on the image's
# writer or the present depends on it) is a compile error: the image cannot
# wait across a host function in one present segment. Read a frame on the
# host through an offscreen target copied to the window.
hostinpresent(p) = throw(ArgumentError("a host node between rendering into the window and presenting it"))
function addnode!(s, c, p::Node)
    mustcut(c, p) && closesegment!(c)
    push!(c.segment, p); c.time += p.compiled.cost
    added!(c, p)
end
mustcut(c, p) = !c.segment.presents &&            # the present segment is not cut before its PresentNode
    (c.segment.kind !== p.compiled.queue || c.time + p.compiled.cost > c.budget || writeswindow(p))
added!(c, p::Node) = writeswindow(p) && (c.segment.presents = true)
added!(c, p::PresentNode) = closesegment!(c)      # the present segment ends here; the next node opens a new one
# The present segment is one submission on one queue: from the first node that
# writes the swapchain image to the PresentNode. presentspan! runs after
# hostcuts! (whose copies may land in the span): every node in the span gets
# the present kind (Main(); the run picks one of its queues that can present
# to the window); a BuildNode or HostNode in it that nothing in the
# span depends on, and that depends on nothing in it, moves before the span;
# one that cannot move is a compile error (a per-image recording holds no
# BuildRecording, and the image cannot wait across a host function).
# A node moves out when it does not depend, directly or transitively, on the
# first window write (judged per node in order, so a node depending only on
# nodes that moved out moves too); everything else stays and gets the
# present kind.
function presentspan!(s)
    span = presentspanof(s)                          # empty without a window
    w = first(span)
    for p in span[2:end]
        dependson(s.dag, p, w) ? setqueue!(p, Main()) : movebefore!(s, p, w)
    end
    any(p -> notinpresent(p), presentspanof(s)) &&
        throw(ArgumentError("a build or host node between rendering into the window and presenting it"))
end
notinpresent(::Union{BuildNode,HostNode}) = true
notinpresent(_) = false

# Expanded loop nodes retain iteration boundaries; core cuts between iterations
# before an estimated budget overrun. No opaque loop remains for a backend to split.

# ── summary!: every access of every stage to what the plan shares ──
function summary!(s)
    for stage in s.stages
        for seg in stage.segments, p in seg.nodes, (r, u) in uses(p, s)
            shared(r) && push!(get!(s.summary, seg, []), (r, u))
        end
        stage.host === nothing && continue
        # A host node's accesses are the host's; they belong to the stage after it.
        stage.hostuses = [(r, u) for (r, u) in uses(stage.host, s) if shared(r)]
    end
    # shared: device arrays (storage and cell), resizable graph arrays (their
    # cell), images, acceleration structures, GPURef slots, arena parts.
    # Bound resources are the plan's own. A composite (a TLAS, a BLAS) is
    # listed as itself; its members are added at run, from the snapshots
    # lockexpanded takes (resizing-and-raytracing.jl B.4), since push! and
    # delete! change them without recompiling.
end

# ── bindings!: the argument block and the tables the head and prepares read ──
function bindings!(s)
    for p in launches(s)
        p.compiled.slots = allocateslots!(s.block, p)       # fields directly in the slots
        for (slot, a) in arrayslots(p)
            bindslot!(s, slot, a, a.state.cell)
        end
    end
    s.residency = residencylist(s)
    s.descriptors = compilebindings(s.device, s)            # device-type verb: descriptor objects
end
bindslot!(s, slot, a, ::NoCell) = write!(s.block, slot, transientaddress(s, a))
bindslot!(s, slot, a, c::Cell) = push!(s.celltable, CellCopy(c, slot, viewof(a), groupsizes(s, a)))

# ── Rewriters: only if opted in; generic, never specific to a device type ──
struct FuseEpilogues end
function (::FuseEpilogues)(s)
    fused = Tuple{Node,Node}[]
    for p in s.nodes
        c = candidate(p)                              # the selected mul!/conv! candidate
        c === nothing && continue
        q = soleconsumer(s.dag, p, output(p))         # the only node reading p's output next
        q !== nothing && q.origin.f === broadcast! || continue
        implements(c, elementwise(q.origin)) && push!(fused, (p, q))   # declared by the candidate, matched exactly
    end
    for (p, q) in fused                               # applied after the scan
        replace!(s.nodes, p => withepilogue(p, elementwise(q.origin))); remove!(s.nodes, q)
    end
end

struct MergeRegions{A}; attribute::A; end
function (r::MergeRegions)(s)
    runs = maximalruns(p -> hasattribute(candidate(p), r.attribute), s.dag)   # no path leaves a run and re-enters it
    for run in runs
        replace!(s.nodes, run => RegionNode(regioncompiler(candidate(first(run))), run))
    end
end

# ══ 5. Record: core walks, the device type writes ═════════════════════════════

function record!(pl::Plan)
    for stage in pl.stages, seg in stage.segments
        windowed(seg) && continue                    # a present or window-sized segment: recordwindows!, per generation
        seg.parts = Any[]
        b = Builder(pl.device, seg.kind)             # command objects of the kind, from the device's free list
        for p in seg.nodes
            b = emitpart!(b, seg, pl, p)             # the builder to continue in
        end
        push!(seg.parts, Recording(finish(b), seg.kind))
    end
end
# Record compiles nothing, allocates nothing, submits nothing, reads no mutable
# resource field: everything it writes comes from the compiled nodes. A
# segment is a list of parts submitted together: recordings, written here
# once, and BuildRecordings. A run re-records only the BuildRecordings a change
# marked (resizing-and-raytracing.jl B.8), the parts that recorded the handle
# of an array that moved (an index buffer), a pass's list
# after a piece insert, delete or sort! and, per swapchain
# generation, the segments that draw into window-sized targets and the
# per-image present segments (section 7).

emitpart!(b, seg, pl, p::Node) = (emit!(b, pl, p); b)
# A build or refit ends the part on every device type: it goes into an
# BuildRecording, its own small command buffer that the first run records with
# the same builder verbs, and that a run records again only when a change
# marked it (a new count, a MoveAccel; B.8). Its barrier is recorded with it.
function emitpart!(b, seg, pl, p::BuildNode)
    push!(seg.parts, Recording(finish(b), seg.kind), BuildRecording(p.structure, p.mode, deps(p), seg.kind))
    return Builder(pl.device, seg.kind)
end

emit!(b, pl, p::KernelNode)  = emitdispatch!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::TraceNode)   = emittrace!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::PrepareNode) = emitdispatch!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::CopyNode)    = emitcopy!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::ForeignNode) = emitlibrary!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::RegionNode)  = emitlibrary!(b, p, p.compiled.command, deps(p))   # a region compiler's output
# Draws inside a render pass carry no barrier: the pass's barrier (target
# layouts, and everything its draws read) is beginrender!'s. The pass is its
# list of pieces, in draw order (below).
function emit!(b, pl, p::RenderNode)
    beginrender!(b, p, p.compiled.command, deps(p))   # the fixed barrier included
    emitpieces!(b, p, [cp.native for cp in piecesof(pl, p)])   # the plan's list (updatepieces!); device-type verb
    endrender!(b)
end

# ── Render passes: formed by compile ──
# Draws and clears are nodes in the order like launches; there is no pass
# object. Core forms the passes: clears and draws into the same targets, in
# order with no node between them that reads what they write, are one
# RenderNode. Other nodes between them are scheduled before or after the pass.
# A target's load operation is Clear(v) when a clear of it comes before its
# draws in the pass and Keep() otherwise; its store operation is Discard()
# only for a graph transient with no later read or Keep() load. Device images
# and windows always Store(): host reads, other graphs and presentation are
# observable even when absent from this graph's node list. The pass's
# draws are cut into pieces by their holder: each run of draws one group (or
# the graph itself) holds is one piece.
mutable struct RenderNode{T<:Tuple,L<:Tuple} <: Node
    const targets::T                        # windows and images: colour, then depth
    const loads::L                          # per target: Clear(v) or Keep()
    pieces::Vector{Piece}                   # in order
    compiled::Union{Nothing,CompiledNode}
end
struct Piece                                # the draws one holder has in one pass
    holder::Union{Graph,Group}
    draws::Vector{DrawNode}                 # the plan's copies, in order
end
function formpasses!(s)
    for run in passruns(s)                  # maximal runs of ClearNodes and DrawNodes into the same targets
        p = RenderNode(targetsof(run), loadsof(run), cutpieces(run), nothing)   # cut by holder
        replacerun!(s.nodes, run, p)        # the nodes between them that touch none of their resources move out
    end
end

# ── Pieces: groups of draws change without a recompile ──
# Pieces serve complex GUIs without slowing large meshes and scatters: an
# insert may take milliseconds, and visibility is cheap. A group whose nodes
# are draws into one pass is a piece of that pass, compiled and recorded on
# its own. On a graph with a plan, a new group of draws, draws declared into
# such a group, its delete! and a sort! of such groups compile and record
# that group's piece only and rewrite the pass's list (one part recorded
# again before its next submission, as a marked BuildRecording); the plan is
# not recompiled and no other piece is recorded again. Each draw keeps its
# own pipeline, index buffer and argument layout, so a large mesh draws
# exactly as it would alone. Visibility stays a store into the draw's when!
# flag (the count times the flag): no piece is touched.
#
# What makes a piece independent of the plan:
#   - its own argument region (a pool region the piece holds) and its own
#     entries in the head's cell-copy table, which is a resizable array of
#     the plan: the head launches over its length (indirect), so new entries
#     record nothing; a deleted piece's entries are blanked and reused, and
#     the table is compacted at the next recompile;
#   - a pass that accepts new pieces has an AnyWrite read in dependencies!
#     and barriers!: it follows every earlier writer and precedes every later
#     writer, on all queues. crossqueuewaits! retains these edges. Its barrier
#     includes every supported producer stage and every draw-input stage;
#   - fast-path draws only read non-attachment resources. A shader storage
#     write, a new image-layout transition, an attachment feedback dependency
#     or a required change of queue/placement recompiles the plan. A memory
#     barrier cannot replace a queue wait or an image-layout transition;
#   - its own registration: the piece's resources join the namers,
#     holds and dependents, as register! does for a plan, for these uses only.
# Core Vulkan inherits no state into a secondary command buffer ("No state,
# including dynamic state, is inherited from one command buffer to
# another", pipelines.adoc:131-133), so each piece binds its pipelines and
# sets viewport and scissor itself: the pieces of a pass into a window are
# recorded per swapchain generation, like the other window-sized parts (a
# resize records each piece once).
#
# Two lists. The graph's groups hold the declared draws (declarations,
# delete! and sort! edit them under the graph lock; a recompile compiles from
# them). Each plan holds its own compiled list per pass (piecesof), which
# only updatepieces! edits, under the graph lock before the run (run!(g)): a
# new piece joins it once compiled and recorded, a deleted one leaves it
# before its region and recording are retired. recordwindows! records the
# plan's list, after updatepieces! (run!(g)).
mutable struct CompiledPiece{N}             # one plan's; released with the plan, or by a delete
    const piece::Piece
    args::Region                            # its argument region (the slot addresses of the GPURefs it reads, baked)
    copies::Vector{Int}                     # its entries in the plan's cell-copy table
    native::N                               # the device type's recording (recordpiece), per generation for a window pass
end

# draw!(g or group, target, pipeline, args, count; instances, indices): one
# DrawNode at the end of its place. when! is one conditional draw: indirect,
# its count multiplied by the Bool flag, as for a when! launch (section 2);
# visibility is a store into the flag. clear! is the load operation of
# target in the pass of the draws after it. On a device each runs now, as its
# own pass (section 11).
draw!(l::Union{Graph,Group}, target, pipeline, args::Tuple, count; instances = 1, indices = nothing) =
    declare!(l, DrawNode(target, pipeline, args, count, instances, indices, nothing, Origin(draw!, args), nothing))
when!(l::Union{Graph,Group}, target, flag::Union{GPURef{Bool},MantleArray{Bool,1}}, pipeline, args::Tuple, count;
      instances = 1, indices = nothing) =
    declare!(l, DrawNode(target, pipeline, args, count, instances, indices, flag, Origin(when!, args), nothing))
clear!(l::Union{Graph,Group}, target::Union{Window,Image}, v) =
    declare!(l, ClearNode(target, v, Origin(clear!, (target, v)), nothing))

# What a declaration or delete! on a graph with a plan leaves for the next run.
struct Added{P<:Union{Graph,Group},N<:Node}
    parent::P; node::N
end
struct Deleted
    group::Group
end
struct Reordered
    group::Group
end

# Applied by run!(g) under the graph lock, before the run's stages. Piece
# changes only: updatepieces!. Anything else: a recompile, which compiles
# every piece anew.
applychanges!(g::Graph) = all(c -> piecechange(g.plan, c), g.plan.changes) ? updatepieces!(g.plan) : recompile!(g)
# A piece change touches only draws in passes the plan has: a group made
# (empty, or of draws whose place is inside a pass of the plan with their
# targets), a draw declared into such a group, a deleted group of draws, a
# sort! of groups of draws in one pass, provided piececompatible verifies the
# read-only inputs, already valid image layouts and placement/queue contract
# above from compiled shader access. Otherwise recompile. A launch, a clear, a trace, a draw
# into the graph itself, a group of draws that would start a new pass: a
# recompile.
piecechange(pl, c::Added) = piecechange(pl, c.parent, c.node)
piecechange(pl, ::Union{Graph,Group}, gr::Group) = gr.loop === nothing && drawsonly(gr) &&
    (isempty(gr.nodes) || (passof(pl, gr) !== nothing && all(d -> piececompatible(pl, gr, d), gr.nodes)))
piecechange(pl, gr::Group, d::DrawNode) = drawsonly(gr) && passof(pl, gr) !== nothing && piececompatible(pl, gr, d)
piecechange(pl, ::Union{Graph,Group}, ::Node) = false
piecechange(pl, c::Deleted) = drawsonly(c.group)
piecechange(pl, c::Reordered) = (gs = filter(isgroup, c.group.nodes); all(drawsonly, gs) && onepass(pl, gs))
drawsonly(gr::Group) = all(isdraw, gr.nodes)
isdraw(::DrawNode) = true
isdraw(::Node) = false
# passof(pl, gr): the plan's RenderNode holding the group's piece; for a
# group the plan has no piece of, the pass its place lies in (the draw or
# clear before or after the place in the order belongs to it) whose targets
# are the targets of the group's draws; nothing otherwise.

# An added or changed group: pipelines from core's device-owned cache, the
# argument region (acquire!, before any lock, like prepare!), the cell-copy
# entries (a store into the plan's table, applied by the run's first stage),
# the piece's recording (recordpiece) and its registration; a group recorded
# before has its old CompiledPiece retired with the plan's last tokens. A
# delete: the group's draws leave the namers (its last-run tokens into the
# resources' `retired`, the other namers notified) and their holds drop, its
# region, entries and recording are retired with the plan's last tokens (the
# graph lock is held, so no run of this plan is submitting). A sort!: nothing
# but the list below. In every case the part holding the pass is marked and
# recorded again before its next submission. A group that fails to compile
# (a pipeline error, memory) is deleted as delete! does, its resources
# unnamed, and run! throws; the rest of the graph keeps running.
function updatepieces!(pl::Plan)
    added, deleted, sorted = Group[], Group[], Group[]
    foreach(c -> collectpiecechanges!(added, deleted, sorted, c), pl.changes)
    unique!(added)
    passes = unique(filter(!isnothing, [passof(pl, gr) for gr in [added; deleted; sorted]]))
    try
        foreach(gr -> deleted!(pl, gr), deleted)
        foreach(gr -> islive(gr) && !isempty(gr.nodes) && recordgroup!(pl, gr), added)
    finally
        empty!(pl.changes)                          # a failed group was deleted; run! throws after this
        for pass in passes                          # the plan's list: the graph's order, compiled pieces only
            setpieces!(pl, pass, compiledpieces(pl, pass))
            markpart!(pl, pass)                     # the list part is recorded again before its next submission
        end
    end
end
collectpiecechanges!(added, deleted, sorted, c::Added) = push!(added, changedgroup(c))
collectpiecechanges!(added, deleted, sorted, c::Deleted) = push!(deleted, c.group)
collectpiecechanges!(added, deleted, sorted, c::Reordered) = append!(sorted, filter(isgroup, c.group.nodes))
changedgroup(c::Added{<:Union{Graph,Group},Group}) = c.node   # a group made
changedgroup(c::Added{Group,<:DrawNode}) = c.parent           # a draw declared into a group
function recordgroup!(pl::Plan, gr::Group)
    old = compiledpiece(pl, gr)                     # nothing for a new group
    ok = false
    try
        cp = compilepiece!(pl, Piece(gr, [deepcopynode(d, gr) for d in gr.nodes]))   # pipelines, region, copy-table entries, recordpiece
        registerpiece!(pl, cp)                      # register! (resizing-and-raytracing.jl B.6) for its uses only;
        ok = true                                   #   a resource the plan names already gets the union of both uses
    finally
        ok || (gr.live = false; filter!(n -> n !== gr, nodelist(gr.parent));
               dropunnamed!(graph(pl), declared(gr), ()))   # never used by a run
    end
    old === nothing || releasepiece!(pl, old, lasttokens(pl))
end
function deleted!(pl::Plan, gr::Group)
    tokens = lasttokens(pl)
    dropunnamed!(graph(pl), declared(gr), tokens; plan = pl)
    cp = compiledpiece(pl, gr)
    cp === nothing || releasepiece!(pl, cp, tokens)   # its CompiledPiece: region, copy-table entries (blanked), recording
end
# The resources no other node names any more (unname! puts them in
# g.unnamed; a recompile ran first if one was due, so nothing else is there):
# the plan leaves their namers (its last-run tokens into their `retired`, the
# other namers notified, as unregister!), and their holds drop after `tokens`.
function dropunnamed!(g::Graph, nodes, tokens; plan = nothing)
    unname!(g, nodes)
    plan === nothing || unregisteruses!(plan, g.unnamed)
    foreach(st -> dropafter!(st, tokens), g.unnamed); empty!(g.unnamed)
end
# A recompile compiles every piece with the new plan. It first unnames the
# nodes of groups deleted since the last run (their holds drop with the old
# plan's tokens, below) and clears the changes.
recompiled!(g, c::Deleted) = unname!(g, declared(c.group))
recompiled!(g, ::Union{Added,Reordered}) = nothing
# A PresentNode records only its barrier (the image to the present layout);
# the present itself is presentnative! after record-lock publication. A VideoDecodeNode
# records through the decode verbs of the plan's video section.
emit!(b, pl, p::PresentNode) = emitcopy!(b, p, CompiledCopy(NoCopy()), deps(p))   # explicit barrier-only copy command
# The head: copies cells into the Bound slots, writes the launch commands and
# ndrange values over resizable arrays, and multiplies each when! launch's
# count by its flag where the host writes the flag (section 2). It writes no
# values: a GPURef is read through its slot. Its barrier to the
# launches after it is theirs (barriers! put it on the first reader of each Bound).
emit!(b, pl, p::HeadNode) =
    for k in p.compiled.command.kernels              # copycells, launches
        emitdispatch!(b, p, k, deps(p))
    end
deps(p) = Deps(p.compiled.preds, p.compiled.barrier)

function copycells_kernel!(slots, cells, table)
    i = KI.get_global_id().x
    i <= length(table) || return
    e = table[i]
    slots[e.slot] = viewed(cells[e.cell], e.view)   # address + static offset, dims and strides of the view
end

# ══ 6. Run ════════════════════════════════════════════════════════════════════

# The graph lock is not reentrant: a host function of this graph that runs,
# frees or writes into the same graph is an error, not a deadlock. The same
# holds for a graph that shares an arena part the calling task holds (a host
# function of another graph on the same memory): it would wait for its own
# run. The part check reads holders without their locks: only the current
# task can make itself a holder, so the answer about itself is exact. (Host
# functions of two graphs that run each other without sharing memory can
# still deadlock, and so can a host function that starts another task which
# runs the same graph and waits for it: both checks are per task. Mantle does
# not detect those.)
function lockgraph(f, g::Graph)
    (@atomic g.owner) === current_task() &&
        throw(ArgumentError("a host function of this graph used the graph itself"))
    holdsany(current_task(), g) &&
        throw(ArgumentError("a host function used a graph that shares arena memory with its own running graph"))
    Base.acquire(g.lock) do
        @atomic g.owner = current_task()
        try f() finally @atomic g.owner = nothing end
    end
end
holdsany(task, g::Graph) = g.plan !== nothing && any(p -> p.holder === task, arenaparts(g.plan))

# run!(pl) returns Submitted(), Skipped() (a minimized window: nothing
# submitted, the changes stay pending) or Again(). Again means nothing was
# claimed: a dependency changed, an arena range or an evicted resource it
# needs changed since it looked, or the window changed under it; run!(g)
# applies the change and runs again. An acquired swapchain image goes back to
# the window, which owns it, and the next attempt takes it from there.
# A WaitFor (stage 1 met a change another plan's run cannot follow mid-run)
# is waited for outside the graph lock and the arena parts (run!(pl)'s finally
# released them), so a host function of that other plan can still run this
# graph; then the run starts over. Windows are applied before the recompile
# check, so a window resized past its capacity recompiles in the same run
# (plan, Run). Pieces are updated before the windows are recorded, so a new
# generation records the current list. run!(g) returns what the plan's run
# returned: Submitted() or Skipped(), both public.
function run!(g::Graph)
    while true
        r = lockgraph(g) do
            (@atomic g.freed) && throw(ArgumentError("run! of a freed graph"))
            while true
                (g.plan !== nothing && hasrenderedpending(g.plan.window)) && return retrypresent!(g.plan.window) # before edits/recompile; presentation only
                applywindows!(g)    # a resize! of a window this graph renders to: generation, cell writes, mark (section 7)
                (g.plan === nothing || @atomic g.recompile) && recompile!(g)
                isempty(g.plan.changes) || applychanges!(g)      # declarations and deletes since the last run (section 5)
                recordwindows!(g)   # the current generation's segments, if the plan has none for it yet (section 7)
                v = run!(g.plan)
                again(v) || return v                # Submitted, Skipped or WaitFor
            end
        end
        done(r) && return r                         # Submitted() or Skipped()
        waitforruns(r)                              # outside every lock; throws inside a host function
    end
end
done(::Union{Submitted,Skipped,Mapped,Registered,Pended,Done,StageDone,Lent}) = true
done(::Union{Tuple,AbstractVector}) = true          # submitwith!'s (token, records); a list of steps
done(::Union{Again,WaitFor}) = false
again(::Again) = true
again(_) = false

# Recompiles happen after nodes were declared into the graph or deleted from
# it, a dependency changed (markstale!, resizing-and-raytracing.jl B.6), a
# graph array or window grew past the size its slot was placed for, or a
# compile failed.
function recompile!(g::Graph)
    @atomic g.recompile = false
    if g.plan !== nothing
        foreach(c -> recompiled!(g, c), g.plan.changes); empty!(g.plan.changes)
    end
    ok = false
    new = nothing
    try
        new = compile(g)            # fallible work, all before any token is claimed
        record!(new)
        while true
            registered = register!(new)
            done(registered) && break
            discard!(new); new = nothing
            waitabletransfer(registered) && waitfortransfers(registered) # no record/arena locks at compile
            new = compile(g); record!(new)
        end
        for (w, sc, size) in currentgenerations(g)   # each captured under its window's lock (section 7)
            recordwindows!(new, w, sc, size)
        end
        old, g.plan = g.plan, new
        old === nothing || (new.lastrun = copy(old.lastrun))   # graph-wide order survives plan replacement, even for disjoint resources
        ok = true                             # new is the graph's plan now: a throw below must not discard it
        old === nothing || (new.compiles = old.compiles + 1)   # carried: the count a test asserts did not move
        old === nothing || retireplan!(old, new)   # B.6: off the dependents and
                                              #   namers, structures' command lists, held targets
        foreach(st -> dropafter!(st, lasttokens(old)), g.unnamed); empty!(g.unnamed)   # resources deleted nodes named
    finally
        # A failed compile leaves the mark; a plan it made is discarded, so no
        # registration in dependents or namers, command list, held target or
        # arena tenancy outlives it.
        ok || (@atomic g.recompile = true; new === nothing || discard!(new))
    end
end
# The old plan after a recompile: unregistered, its tenancy dropped in the
# arena parts the new plan does not use (the graph stays a tenant of those it
# shares, with the new plan's peak), retired with its last tokens.
retireplan!(old, new) = (unregister!(old);
    gone = setdiff(arenaparts(old), arenaparts(new));
    locked(() -> foreach(a -> untenant!(a, old.graphid), gone), [a.sync for a in gone]; lent = true);
    retire!(pool(old.device), old, lasttokens(old)))
# A plan that never ran: unregister! (nothing if it never registered), its
# tenancy in every arena part dropped (untenant!, under the parts' locks), then retired.
discard!(pl::Plan) = (unregister!(pl);
    locked(() -> foreach(a -> untenant!(a, pl.graphid), arenaparts(pl)), [a.sync for a in arenaparts(pl)]; lent = true);
    retire!(pool(pl.device), pl, ()))

# What a run allocated or acquired before taking its locks. A change that
# uses one of them takes it out of here under the locks; whatever is still
# here when the run ends (Again, a throw, storage restored meanwhile by
# another run) goes back in `finally`. The same rule as every change
# (resizing-and-raytracing.jl B.4, change!).
mutable struct RunState
    pages::IdDict{Arena,Vector{PageRun}}   # per arena part of the plan: pages for its unmapped range (section 9)
    present::Union{Nothing,PendingPresent}
    prepared::Prepared                  # what applying the pending changes and restores needs (B.4 step 1)
    image::Union{Nothing,Image,WindowChanged,Minimized}
    generation::Union{Nothing,Swapchain} # the swapchain generation this run's window-sized segments use
    waits::Vector{Token}                # other graphs' work found by earlier stages (section 8); later stages wait too
    channels::Dict{QueueKind,SubmitChannel}   # the run's pick per queue kind, made before its first submission (section 8)
end

# Two flags. `running` (set at the start, cleared at the end) keeps
# unmapidle! and eviction off what the plan names. `between` is set under
# stage 1's locks once the stale check has passed, and cleared when the last
# stage is submitted, under pl.runend's lock with a notify: only while it is
# set could a change the plan cannot follow mid-run reach a later stage, so
# only then does another submitter applying such a change wait
# (resizing-and-raytracing.jl B.4); the plan's own later stages leave it for
# after the run. Before stage 1, the stale flag makes the run start over, so
# nothing waits on a run still waiting for its arena parts. A plan with one
# stage clears it inside the same locked section: nothing ever waits on it.
#
# Each stage applies the pending changes of what it locks (B.4): before its
# locks it prepares their storage (prepare!, B.4 step 1: storage, pages,
# capacity, command objects for the BuildRecordings it will record again, and
# restore storage for evicted resources when the plan is marked for restore;
# a store's upload region was acquired at its call), then under them
# submitwith! applies them. If what
# was prepared no longer fits (a change arrived meanwhile, during a host
# function say), the stage prepares again and retries; nothing of the stage
# was submitted, and earlier stages stay as they were.
function run!(pl::Plan)
    sweep!(pool(pl.device))         # release what has passed (section 9)
    @atomic pl.running = true       # unmapidle! and eviction leave what this plan names alone until the run ends
    rs = RunState(IdDict(), nothing, Prepared(), nothing, nothing, Token[])
    held = false
    try
        # The image before the arena parts: its acquire may block (vsync, a
        # hidden window), and no other graph's run waits on this run's parts
        # meanwhile.
        rs.image = acquireimage!(pl)     # before any lock (section 7)
        skip(rs.image) && return Skipped()
        again(rs.image) && return Again()
        if renderedpending(rs.image)
            # Previous rendering succeeded but present was not enqueued (OOM).
            # Retry ONLY that present; do not replay device-side sample counters.
            rs.generation = generationof(rs.image)
            rs.present = pendingpresent(rs.image)
            finishpresent!(rs)
            return Submitted()
        end
        holdparts!(pl); held = true # before any lock: waits while another run holds a shared arena part
        acquireunmapped!(rs, pl)    # per arena part, fallible, so before any lock (section 9)
        # The image's generation: held by the image until the present, then
        # by the run until its last stage is submitted (window-sized segments
        # after the present and on other queues use its recordings). Recorded
        # now if the plan has none for it yet (the window was resized after
        # applywindows!; graph lock held, no record lock).
        rs.generation = generationof(rs.image)
        rs.generation === nothing || recordedfor(pl, rs.generation) ||
            recordwindows!(pl, pl.window, rs.generation, rs.generation.size)
        for (i, stage) in enumerate(pl.stages)
            i > 1 && runhost!(pl.stages[i-1])        # waits for its inputs, calls the host function
            # A stage locks only what its inbox names (resources written by
            # other runs, changed or accessed eagerly since the plan's last
            # look), the evicted resources the plan names when it is marked
            # for restore (stage 1), and its arena parts (stage 1), plus the
            # sync locks of its plan, of the plans sharing what the stage
            # writes and of the plans naming those resources (section 8).
            # Nothing else is looked at.
            r = nothing
            while true
                snap = inboxsnapshot(pl)             # lock-free
                extra = i == 1 && (@atomic pl.restore) ? evictednamed(pl) : ()   # walked only when marked
                prepare!(rs.prepared, [snap; extra]; restores = !isempty(extra))
                r = lockstage(pl, i, snap, extra) do recs, items
                    if i == 1
                        (@atomic pl.stale) && return Again()  # a dependency changed (B.6): recompile first
                        done(bindunmapped!(pl, rs)) || return Again()
                    end
                    r = submitstage!(pl, stage, recs, items, rs)  # Again or WaitFor from submitwith!: nothing submitted
                    done(r) && i == 1 && (@atomic pl.between = true; @atomic pl.restore = false)
                    done(r) && i == length(pl.stages) && endbetween!(pl)   # still under the locks
                    r
                end
                if done(r)
                    break
                end
                i == 1 && return r                   # Again or WaitFor: run!(g) starts over (waiting outside its lock)
                # Stale-making changes stay deferred after stage 1. An internal
                # copy transfer may still be busy: wait outside record locks,
                # then retry THIS stage, preserving previous accepted stages.
                waitabletransfer(r) && waitfortransfers(r)
            end
        end
        finishpresent!(rs)                   # ALL stages accepted; outside plan/resource locks
        return Submitted()
    finally
        endbetween!(pl)                               # a throw between stages; nothing if already cleared
        foreach(((part, pages),) -> release!(pool(pl.device), pages), rs.pages)   # pages no change took
        release!(rs.prepared)                         # what no step took
        abandonunattemptedpresent!(rs)                # partial failure: retire signal semaphore; do not mark complete-frame retry
        keepimage!(pl.window, rs.image)               # an image not presented goes back to its owner, the window
        # A presented image's generation is released with every token this
        # run submitted; an image not presented goes back to the window with
        # its hold (keepimage!, above).
        rs.generation === nothing || rs.image !== nothing || releasegeneration!(rs.generation, lastrun(pl))
        held && releaseparts!(pl)
        @atomic pl.running = false
    end
end
# The locks of one stage (section 8, Locks): the sync locks of the plan, of
# the plans sharing what the stage writes and of the plans naming the inbox
# (and restore) resources, expanded to the members a pending build of theirs
# reads (their writers' last runs are read for the build's waits), then the
# SyncStates of those resources and, for stage 1, of the arena parts. Under
# them: if more arrived in the inbox, or a naming snapshot now has a plan not
# locked (a plan registered, a composite gained a member), the stage starts
# over; nothing recompiles. The stage handles exactly the snapshot it locked;
# its items leave the inbox only once the stage has submitted (dropinbox!
# removes those items and nothing pushed meanwhile). A stage that returns
# Again or WaitFor leaves them, so its retry, or the next run, handles them.
function lockstage(f, pl, i, snap, extra)
    stage = pl.stages[i]
    res = expanded([snap; extra; writtenwithreaders(pl.stages[i])])
    # Include written arrays and their transitive readers, from immutable
    # membership snapshots checked below. Updates must be pended before these
    # locks are released, so another run cannot observe new geometry without
    # the corresponding pending BLAS/TLAS update.
    plans = lockset(pl, stage, res)
    r = lockplans(plans) do
        recs = [[syncstate(r) for r in res]; (i == 1 ? [a.sync for a in arenaparts(pl)] : []);
                [c.structure.sync for c in accelcommands(stage)]]   # what current! reads (B.8)
        locked(recs) do
            inboxsnapshot(pl) === snap || return Retry()
            lockset(pl, stage, res) ⊆ plans || return Retry()
            readerscurrent(res) || return Retry()
            busy = transferwaits(recs, nothing)
            isempty(busy) || return Done(WaitForTransfers(busy)) # before any stage segment
            v = f(recs, snap)
            done(v) && dropinbox!(pl, snap)  # lock-free: a compare-and-swap that keeps later pushes
            Done(v)
        end
    end
    return done(r) ? r.value : Again()       # Retry: the stage's loop prepares and locks again
end
lockset(pl, stage, res) = unique([pl; sharers(pl, stage); (p for r in res for p in namingplans(r))...])
# The items and the members their pending BuildAccels read (uses, B.3), as
# lockexpanded does for a change.
expanded(items) = unique([items; (m for r in items for c in pendingof(r) for (m, _) in uses(c))...])
endbetween!(pl) = lock(pl.runend) do
    @atomic pl.between = false
    notify(pl.runend)
end

# One stage's submissions, under its locks. The waits on other work are
# computed before any access is recorded, so the stage's own segments do not
# wait on each other through a shared record; their order inside the run
# comes from crossqueuewaits! (seg.after: segments of this run, read when the
# segment is submitted, so their tokens are this run's).
function submitstage!(pl, stage, recs, items, rs)
    # Waits: other graphs' work on what the stage writes and on what its
    # inbox brought (section 8), added to the run's and kept for its later
    # stages, and this plan's previous run; every segment of the stage gets
    # them (a segment on another queue may have no path from the first).
    # The run's own accesses are recorded nowhere but in its last-run tokens,
    # set here under the plan's sync lock.
    rs.waits = latestperchannel(filter(!passed, [rs.waits; otherwaits(pl, stage, items)]))
    w = copy(rs.waits)
    stage === first(pl.stages) && append!(w, lastrun(pl))   # runs never overlap
    # One queue per kind for the whole run, picked from its waits under its
    # locks before its first submission (section 8); its later stages use them.
    stage === first(pl.stages) && (rs.channels = pickchannels(pl, w))
    waits = Dict(seg => w for seg in stage.segments)
    # The first segment carries the pending changes of the inbox resources, in
    # front of its head node (resizing-and-raytracing.jl B.4), applied with
    # what prepare! allocated; their steps are recorded as eager accesses and
    # tell the other plans naming those resources. A change this plan cannot
    # follow mid-run stays pending for after the run (self = pl). If
    # submitwith! returns Again or WaitFor, nothing of the stage was
    # submitted: the stage starts over (run!).
    changed = ()
    for (k, seg) in enumerate(stage.segments)
        present = presentof(rs.image, seg)
        parts = segparts(seg, rs)                     # the present segment: the acquired image's recording;
                                                      #   a window-sized one: rs.generation's; others: their own
        ch = rs.channels[seg.kind]
        ownwaits = [waits[seg]; [s.token for s in seg.after]]
        # The parts' command buffers are taken only after the change prefix is
        # emitted (inside submitwith! for the first segment), so a marked
        # BuildRecording is recorded again with the builtcount the prefix's
        # Build just set (B.8).
        if k == 1
            r = submitwith!(ch, recs, rs.prepared, (), parts; self = pl, ownwaits, present,
                            restores = stage === first(pl.stages) && (@atomic pl.restore))
            done(r) || return r                       # Again or WaitFor
            seg.token, changed = r
        else
            touched(seg, changed) && push!(ownwaits, first(stage.segments).token)   # reads what the changes wrote
            seg.token = submit!(ch, map(p -> commandbuffer(p, ()), parts); waits = ownwaits, present)
        end
        # All publication below uses pre-reserved bookkeeping capacity.
        # Do this per accepted segment: a later preparation/native enqueue may fail.
        pl.lastrun[channelstate(ch)] = seg.token
        foreach(p -> submitted!(p, seg.token), parts)
        foreach(r -> notifyreaders!(r, pl), namedwrites(seg))
        pendreaders_locked!(pl, writtenresources(seg))
        present === nothing || (rs.present = PendingPresent(present, seg.token))
        # Native presentation is deferred until ALL run stages have been accepted,
        # outside record locks. The PresentNode only finalizes the image layout.
    end
    return StageDone(writtenresources(stage))
end

# Arena parts are shared between graphs: the memory is the largest tenant's,
# not the sum. A run holds its parts from before its first submission until
# its last stage is submitted, so another graph's run on the same part waits
# on the host for the whole run, host functions included; the GPU order after
# that comes from the tokens. Waiting is preferred over memory of its own.
# Parts are taken in id order, so two runs never wait on
# each other. A task that already holds a part (a host function of a running
# graph) and would wait for a part with a lower or equal id throws instead:
# that wait breaks the id order and can deadlock (with the same part, it
# would wait for itself). lockgraph throws earlier for a graph sharing a part
# the task holds.
function holdparts!(pl::Plan)
    taken = Arena[]
    ok = false
    try
        for part in sort(arenaparts(pl); by = p -> p.sync.id)
            lock(part.cond) do
                while part.holder !== nothing
                    holdsfrom(current_task(), part) &&
                        throw(ArgumentError("a host function runs a graph that could wait on arena memory its own run holds"))
                    wait(part.cond)
                end
                part.holder = current_task()
            end
            push!(taken, part)
        end
        ok = true
    finally
        ok || foreach(releasepart!, taken)   # a throw leaves no part held
    end
end
# Read without the other parts' locks: only this task can make itself a holder.
holdsfrom(task, part) = any(p -> p.holder === task && p.sync.id >= part.sync.id, pool(part).arenas)
releaseparts!(pl::Plan) = foreach(releasepart!, arenaparts(pl))
releasepart!(part::Arena) = lock(part.cond) do
    part.holder = nothing
    notify(part.cond)
end

# Each accepted segment that wrote arrays something else is built from pends
# the readers’ updates before another segment can fail: a TLAS Update,
# a BLAS Update (a Build when the array is its index array), applied by the
# next submission that touches the structure, which waits for this stage's
# write through the sync states (B.4); a screen's redraw mark. `written`
# is the arrays the stage wrote, from its expanded uses. Readers come from each array's snapshot
# (immutable, replaced under its lock; resizing-and-raytracing.jl B.2), so a
# structure built after compile counts too, and iterating needs no lock. A
# structure this plan builds or refits itself has its own command and is
# skipped. One pendupdate_locked! method per reader type appends its requests
# under the stage's already expanded locks.
pendreaders_locked!(pl, written) =
    for a in written, r in readers(a)
        r in builtby(pl) || pendupdate_locked!(r, a)
    end
# pendupdate_locked! appends/coalesces the request and notifies naming plans,
# using the transitive reader locks already held by lockstage. It takes no
# new lock and allocates no device storage. Non-run writers use the same
# publication rule in submitwith!: writes and their reader updates become
# visible atomically under the expanded record locks.
readers(st::ArrayState) = st.readers   # the snapshot
readers(x) = ()                        # a structure, an image, an arena part: nothing is built from them here

commandbuffer(cmd, steps) = cmd                     # a finished command buffer (an eager call, submitnow!)
commandbuffer(p::Recording, steps) = p.native
commandbuffer(c::BuildRecording, steps) = current!(c, steps)   # from the state after `steps`, under the structure's lock (B.8)
submitted!(::Recording, token) = nothing
submitted!(c::BuildRecording, token) = (c.lasttoken = token; built!(c.structure, c.recordedmode, c))   # a Build sets builtcount (B.8)

# A run with host nodes is several submissions. Each stage starts with a head
# node that copies the cells again, so a change made while a host function
# runs (a resize!, a store) is applied in front of the next stage and seen by
# it, unless it made the plan stale (then the next run sees it). The run holds
# its arena parts across the host functions (holdparts!), and unmapidle! skips
# a running plan, so nothing else touches its transients in between.

function runhost!(stage)
    waitfor(tokens(stage))          # the host function's inputs are written
    g = graph(stage)
    @atomic g.hostcaller = current_task()
    try
        stage.host.f(hostviews(stage.host.args)...)   # host-visible arena memory only
    finally
        @atomic g.hostcaller = nothing
    end
end

lastrun(pl) = values(pl.lastrun)   # Tokens, the latest per channel

# ══ 7. Changes ═══════════════════════════════════════════════════════════════
# Stores, resize!, TLAS changes and how a submission applies them:
# resizing-and-raytracing.jl B.3-B.8. Here: GPURefs, windows.

# GPURef(dev, v) for a value: ONE slot in device memory whose
# address every plan and piece that reads it bakes; a launch reads the value
# through it (a one-element buffer read costs what a true uniform costs). The slot is
# a shared resource with a sync state, like a device array: plans name it,
# stand in its namers, and are ordered against the stores into it the same way
# (section 8). Values only: an array argument is the array itself.
mutable struct GPURef{T} <: Resource
    const device::Device
    const slot::Region                  # sizeof(T) bytes of device memory; it never moves, so its address is baked
    const sync::SyncState               # pending stores, eager accesses, namers, the freed mark
    const holds::Threads.Atomic{Int}    # the creator and each graph naming it (section 10)
end
syncstate(r::GPURef) = r.sync
# r[] = v is a scheduled store like any other: the
# value is copied once at the call into an upload region, so a multi-word
# value is never torn, and a Store into the slot is pended through change!
# (B.3's Store, the slot as its storage). The next submission that touches
# the slot applies it in front of its own work (inline, emitstore!, within
# the device type's limit). It marks no plan, and nothing recompiles.
function Base.setindex!(r::GPURef{T}, v) where {T}
    up = acquireupload!(pool(r.device), sizeof(T))    # host-visible system memory, before any lock
    ok = false
    try
        hostview(up, T)[1] = convert(T, v)   # the one copy; a conversion error returns the region
        change!(((r, Write()),)) do
            usable(r)                                 # freed through the creator's handle: throws, under the lock
            Change[Store(nextseq!(r), r, up, 1:sizeof(T))]
        end
        ok = true
    finally
        ok || retire!(pool(r.device), up, ())         # a throw under the lock: the region goes back
    end
    return r
end
# Data the host supplies per run is a device array the graph names, written by
# stores the same way; setindex! on a graph array is nodata() (section 2).

# Window-sized arrays take any dims within the size their slots were placed
# for (the window's capacity: the largest monitor's size unless given); beyond
# it the graph recompiles at its next run. Their contents after a resize are
# undefined. Window-sized images are created again per swapchain generation,
# at the window's size, on the same arena memory: the arena part holding
# window-sized transients is sparse where the device has sparse residency,
# and a generation maps only the pages its size needs, so the capacity costs
# address space, not memory. Without sparse residency the capacity is mapped.

# One graph presents to a window:
# the first graph declaring any write into it (a draw, a clear, a kernel, a copy)
# presents; a second graph declaring one throws at declaration (other graphs
# render into Images this graph reads). The binding ends when the graph is
# freed or collected or the nodes writing the window are removed; another graph may then
# present. An eager draw into a window throws. So every
# acquire and present of a window happens under its graph's lock, in that
# graph's runs: no acquire is ever in progress while applywindows! runs, no
# other graph's run holds one of its images, and a graph has one pl.window.

# The window's resize event calls this. It stores the size and marks the
# window's graph (held weakly: a window keeps no graph alive); nothing else
# happens on the event thread.
function Base.resize!(w::Window, width, height)
    lock(w.lock) do
        w.pendingsize = (width, height)              # beyond the capacity: applywindows! marks a recompile
        g = w.graph.value
        g === nothing || (@atomic g.windowchanged = true)
    end
    return w
end

# In run!(g), under the graph lock, before compile and before any record lock.
# A swapchain generation is a resource with holds: one from the window while
# it is current, one from a run that acquired an image from it and has not
# yet submitted its last stage (the image until the present, the run after
# it; window-sized segments after the present use the generation's
# recordings). Present requests retain independent completion holds
# (execution-contracts.md §4); rendering tokens alone never destroy a generation. A new generation retires the current swapchain (created with it as
# oldSwapchain); an image a failed run kept of the old generation is dropped
# with the generation (a destroyed swapchain releases its acquired images;
# presenting it would draw the old size). A zero size (minimized) makes no
# generation (a zero extent is invalid): the size stays pending and the runs
# skip. A throw restores the mark.
function applywindows!(g::Graph)
    (@atomicswap g.windowchanged = false) || return
    ok = false
    try
        w = window(g)
        w === nothing && (ok = true; return)
        size = lock(w.lock) do
            w.pendingsize === nothing || any(iszero, w.pendingsize) || newgeneration!(w)
            w.size
        end
        foreach(x -> resizeheld!(x, size), windowsized(g, w))  # pending cell writes, or a recompile mark (graph lock held)
        ok = true
    finally
        ok || (@atomic g.windowchanged = true)
    end
end
# In run!(g), after the recompile check and updatepieces!, so a window pass
# records the current list of pieces: this generation's recordings, if the
# plan has none for it yet (a new generation from applywindows!; recompile!
# records the current ones for a new plan). They are the window-sized images
# at this size, the segments that draw into them (render areas and image
# handles are recorded by value), and the per-image present segments. The
# previous generation's are retired with its holds. recordwindows! also
# registers the BuildRecordings of those segments in their structures' command
# lists (each under its structure's lock, as register! does), so a change
# marks them, and removes the retired generation's; and it sets the plan's
# tenancy in the window part for this size, marking the part when it grew
# (section 9).
function recordwindows!(g::Graph)
    w = window(g)
    w === nothing && return
    sc, size = lock(() -> (w.swapchain, w.size), w.lock)
    recordedfor(g.plan, sc) || recordwindows!(g.plan, w, sc, size)
end
function newgeneration!(w::Window)   # under w.lock; no acquire in progress (the graph lock is held)
    old = w.swapchain
    size = w.pendingsize
    # Native creation can fail and can retire oldSwapchain even on failure.
    # Publish size/current only on success; record old as retired if native
    # creation retired it, so no retry acquires from a retired swapchain.
    result = try
        createswapchain(w, size...; oldswapchain = nativeof(old))
    catch err
        if oldretired(err)
            w.swapchain = nothing
            retireoldgeneration!(w, old) # drop kept/current holds; preserve all completion records
        end
        rethrow()                       # pending size and changed mark remain for retry
    end
    w.swapchain, w.size, w.pendingsize = result, size, nothing
    foreach(i -> drophold!(i.swapchain), w.kept); empty!(w.kept)   # kept images of the old generation go with it
    old === nothing || drophold!(old) # GPU and presentation retirement holds remain independent
end

# In run!(pl), before any record lock. An image a failed run of this graph
# kept (always of the current generation: newgeneration! dropped older ones)
# comes first and is presented by this run. Otherwise, under the window lock:
# a zero extent (minimized) returns Minimized() (the run submits nothing, its
# changes stay pending); a mark set after this run's applywindows! (a resize
# event in between) returns WindowChanged() (Again: the change gets applied).
# The generation's hold is taken under the window lock; the acquire, which
# can block, is not (it blocks only until the driver has an image free: this
# graph's earlier presents, nothing else, hold the window's images).
# OUT_OF_DATE is a resize to the surface's current size, and Again.
function acquireimage!(pl::Plan)
    w = pl.window
    w === nothing && return nothing
    r = lock(w.lock) do
        kept = takeimage!(w)         # of the current generation, or nothing
        kept === nothing || return kept
        any(iszero, w.size) && return Minimized()
        (@atomic graphof(pl).windowchanged) && return WindowChanged()   # graphof: the run holds the graph
        addhold!(w.swapchain)        # the image's hold on its generation (run!'s finally releases it)
        w.swapchain
    end
    return acquirefrom!(w, r)
end
acquirefrom!(w, r::Union{Image,WindowChanged,Minimized}) = r
function acquirefrom!(w, sc::Swapchain)
    image = nothing
    try
        result, index, acquired = acquirenext(sc)    # a binary semaphore on Vulkan; may block, no lock held
        if outofdate(result)
            resize!(w, surfacesize(w)...)
            return WindowChanged()
        end
        image = Image(sc, index, acquired)           # SUBOPTIMAL: presented, and the window marked
        return image
    finally
        image === nothing && drophold!(sc)           # out of date, or a throw: the run holds no generation
    end
end
again(::WindowChanged) = true
again(_) = false
skip(::Minimized) = true
skip(_) = false
# This releases the RUN hold only. Each enqueued present has its own hold and
# independent completion record; tokens alone never destroy the generation.
releasegeneration!(sc::Swapchain, tokens) = retirehold!(sc, tokens)
function finishpresent!(rs)
    rs.present === nothing && return
    req = rs.present
    outcome = try
        r = lock(channelstate(req.channel).lock) do
            presentnative!(req.channel, req.native) # may block in WSI; no record/window lock
        end
        # Retention/ownership publication has pre-reserved storage. A failure
        # here is terminal too: native presentation may already be accepted.
        if r.enqueued
            retainpresent!(req.generation, nativeowner(req.channel, r.retirement))
            rs.image = nothing                    # acquisition consumed, even on OUT_OF_DATE
        elseif presentationretry(r)
            keeprendered!(rs.image, req)            # not-enqueued OOM: retry present only
        else
            releaseunpresented!(req)                # token-safe acquired release
            rs.image = nothing
        end
        rs.present = nothing
        r
    catch err
        poison!(channelstate(req.channel).domain, err)
        quarantinepresent!(channelstate(req.channel).domain, req)
        rs.image = nothing                        # cannot return ambiguous acquisition
        rs.present = nothing                      # quarantine owns it; finally cannot recycle
        rethrow()
    end
    handlepresentstatus!(req.window, outcome.status) # resize/closed mark or deferred error
end
# presentationretry accepts only !enqueued with PresentHostOOM/PresentDeviceOOM.
# handlepresentstatus! throws PresentationRetry for those statuses AFTER keeping
# the request; the next run!(g) retries before edits. Other errors retain their
# accepted GPU prefix. abandonunattemptedpresent! never frees live semaphore uses.
nativeowner(ch, record) = record               # test wrappers retain their owner here
generationof(i::Image) = i.swapchain
generationof(_) = nothing                 # no window

# The window owns an acquired image that no run presented (an Again, a throw):
# kept under w.lock, taken by the graph's next run before it acquires. The
# image keeps its hold on its generation; newgeneration! drops kept images
# with the generation they belong to.
keepimage!(w::Window, i::Image) = lock(() -> push!(w.kept, i), w.lock)
keepimage!(w, _) = nothing           # no window, no image, WindowChanged(), Minimized(): nothing to keep

# ══ 8. Ordering across graphs and queues ══════════════════════════════════════

# A token names one submission: its channel and the value the channel's
# counter reaches when it completes. Core's one-argument forms wrap the
# device type's verbs, which take the channel and the value.
# Native endpoint aliases share state. Tokens retain their endpoint for native
# dispatch, but equality/hash and latest-per-channel use state identity.
abstract type SubmitChannel end
mutable struct ChannelState
    lock::ReentrantLock
    next::UInt64
    domain::FailureDomain   # shared by ALL channels of the same native device
    timeline               # native completion source; one per physical queue
end
struct Token
    state::ChannelState
    channel::SubmitChannel
    value::UInt64
end
Token(ch::SubmitChannel, value) = Token(channelstate(ch), ch, value)
Base.:(==)(a::Token, b::Token) = a.state === b.state && a.value == b.value
Base.hash(t::Token, h::UInt) = hash((objectid(t.state), t.value), h)
passed(t::Token) = (checkhealthy(t.state.domain); passed(t.channel, t.value))    # device-type verb: passed(ch, value)
# A wait gives up two job timeouts after the submission before `t` on its
# channel passed: by then the watchdog has killed it, whether or not the driver
# says so, and the device counts as lost. Earlier submissions get the same
# deadline, each from its own predecessor; with jobtimeout = Inf nothing times out.
function waitfor(t::Token)
    checkhealthy(t.state.domain)
    limit = 2 * caps(device(t.channel)).jobtimeout
    for u in outstanding(t.channel, t)              # the channel's unpassed submissions up to and including t
        waitfor(t.channel, u.value, limit) ||      # device-type verb: waitfor(ch, value, timeout) -> Bool
            lost!(t.state.domain, t.channel)        # poisons the domain; throws DeviceLost
    end
end
# DeviceLost lists the launches of each channel's last submissions (a short
# ring kept at submit) with their estimates and the job timeout: drivers
# report a watchdog kill late (RADV at the next submission), so the error
# names candidates rather than one culprit.
waitfor(ts) = foreach(waitfor, ts)               # a collection of tokens
# A host wait is a driver call made with `@ccall gc_safe = true` (Julia 1.12,
# base/c.jl:278), so a thread waiting on the GPU does not hold up a GC another
# thread needs.

# ── Queue choice: at submission, from what the submission waits on ──
# Compile fixed each segment's queue kind (section 4); the queue of that kind
# is picked when the work is submitted: once per run, under its locks before
# its first submission (all its stages use the pick), and once per eager
# submission, under its locks, before it records. Every pick is correct, since
# every wait names its channel; the pick decides only overlap and how many
# waits cross queues. No task, thread or "current" queue takes part: two tasks
# running independent work land on different queues because the first
# submission's queue is busy when the second one picks. Two picks at the same
# instant may both see one queue idle and share it: that costs overlap, never
# correctness.
pickchannels(pl, waits) = Dict(k => pickchannel(pl.device, k, priority(pl), waits) for k in segmentkinds(pl))
pickchannel(dev, need, priority, waits) = pickchannel(queuepolicy(dev), channels(dev, need, priority), waits)
function pickchannel(::DefaultQueues, chs, waits)
    i = findfirst(ch -> any(t -> t.state === channelstate(ch), waits), chs)
    i === nothing || return chs[i]          # it waits there anyway: no wait crosses queues; a graph waits on its
                                            #   previous run, so it stays on its queue
    i = findfirst(isidle, chs)              # independent work: an idle queue, the main one first
    i === nothing || return chs[i]
    return argmin(outstanding, chs)         # else the least busy
end
pickchannel(p::SpreadQueues, chs, waits) = chs[mod1(Threads.atomic_add!(p.next, 1), length(chs))]
struct DefaultQueues end
struct SpreadQueues                         # the specification's: every submission on the next queue of its kind
    next::Threads.Atomic{Int}
end
SpreadQueues() = SpreadQueues(Threads.Atomic{Int}(0))
# channels(dev, need, priority): the queues that can run the work, main queue
# first. `need` is a recording's QueueKind (a run: exactly that kind, since the
# recording belongs to it) or what an eager submission's work needs, recorded
# after the pick: ComputeWork() (Main and Compute queues), GraphicsWork() (Main
# queues), CopyWork() (every queue). On a device with queue priorities, only
# the queues of the priority's class. How many queues of each kind core
# opens is core's policy, from queuekinds(dev). isidle(ch): every submission
# on ch passed (`passed` on its `next`, non-blocking); outstanding(ch): the
# submissions not yet passed. queuepolicy(dev) and queuepolicy!(dev, p): the
# device's policy, a field of its core state (api.md 2.1).

# ── Between graphs: through the resource, when it is written ──
# No plan keeps a list of other plans. A resource's SyncState has `namers`:
# the plans naming it with their use from compile (read, write or both), an
# immutable snapshot replaced under its lock by register! and unregister!
# (resizing-and-raytracing.jl B.6). A plan naming a composite reads the
# composite's members: namingplanuses(r) is r's namers and, as readers, the
# plans naming the composites built from r (its readers snapshot, B.2).
# Snapshot reads, no lock.
#
# A stage's waits on other graphs, computed under its locks (lockstage):
#   - a resource the stage writes (`namedwrites`, from compile: named arrays,
#     structures it builds, for stage 1 its arena parts) that another plan
#     names: the complete last run of every other plan naming it (write after
#     read, write after write). Whether it is shared is read from the snapshot
#     at run, so a plan registering later needs no recompile. After the
#     stage is submitted, still under the locks, the resource goes into the
#     inboxes of the other plans that read it (notifyreaders!);
#   - a resource in the stage's inbox (written by another run, changed, or
#     accessed eagerly since the plan's last look): waitsfor, i.e. its eager
#     accesses, the last tokens of retired plans that named it, and the last
#     runs of the other plans writing it (read after write).
# Both wait for the other plan's whole last run. The waits are kept in the run's state and every
# later stage waits for them too, so a resource one stage took from the inbox
# and a later stage uses is covered. A stage that writes nothing another
# plan names and finds its inbox empty pays its own sync lock and one
# snapshot read per named resource it writes (its outputs, mostly), nothing
# per resource it only reads.
#
# Why this is race-free: whatever reads another plan's last run holds that
# plan's sync lock, and the snapshots that chose the lock set are checked
# under the locks (Retry; nothing recompiles). Each pair of a write and an
# access by another plan is then ordered one way or the other: the writer's
# stage locks the reader first (and waits for its last run) or the reader's
# stage, holding the writer's lock, finds the write in its inbox (and waits
# for the writer's last run). A plan that starts naming a resource does not
# read it before a submission that holds the sync locks of the resource's
# writers: its first run (register! puts the shared resources it names into
# its own inbox), or, for a composite's new member, the step applying the
# composite's Build (the member is one of the build's uses; lockstage expands
# to it). A plan that stops reading a resource through a composite (delete!)
# stays among its namers until the submission that removed it has
# passed (resizing-and-raytracing.jl B.8); until then writers wait for it.
# A retired plan's last run is in the `retired` list of every resource it
# named, members of the composites it named included, and those resources
# are put into the other namers' inboxes (unregister!).
# A structure or a screen built from r only reads it.
namingplanuses(r) = [namers(syncstate(r)); [(pl, Read()) for c in readers(r) for (pl, _) in namingplanuses(c)]]
namingplans(r) = unique(first.(namingplanuses(r)))
readingplans(r) = unique(pl for (pl, u) in namingplanuses(r) if reads(u))
sharers(pl, stage) = unique(q for r in namedwrites(stage) for q in namingplans(r) if q !== pl)
# namedwrites(stage): from compile.

# Waits of a stage on other graphs. The caller holds the sync locks of
# lockset (lockstage) and the SyncStates of the inbox items. An inbox item
# the stage does not use is waited for with the plan's use, for its later
# stages.
otherwaits(pl, stage, items) =
    [[t for r in namedwrites(stage) for q in namingplans(r) if q !== pl for t in lastrun(q)];
     waitsfor([(r, planuse(pl, r)) for r in items]; self = pl)]

# ── With the host: SyncState and the inbox ──
# What is not a run (eager calls, host reads, vendor views, eviction copies,
# and the steps a submission applies for pending changes) records its
# accesses on the resources' SyncStates. Pending a change (at the call) or
# recording an eager access pushes the resource into the inbox of every plan
# naming it (namingplans: composites built from it included): a lock-free
# list. A run's write of a shared resource pushes it into the inboxes of the
# other plans reading it. A run takes its inbox under its sync lock and
# handles only what is in it: their pending changes are applied in front of
# the stage, the other work on them waited for. Nothing walks the resources
# a plan names at run: a run's host time grows with the shared resources its
# stages write and with its inbox, not with the number of resources it names.
notifyplans!(r) = foreach(pl -> pushinbox!(pl, r), namingplans(r))
notifyreaders!(r, self) = foreach(pl -> pl === self || pushinbox!(pl, r), readingplans(r))

# Waits of a submission that is not a run, and of the steps a run's prefix
# applies (they write what they change): for each resource, its eager
# accesses, the last tokens of plans retired while naming it, and the last
# runs of the plans that name it, where their use conflicts (each plan's use
# of a resource, read or write, is known from compile). The caller holds
# those plans' sync locks (their last runs cannot change under it) and the
# resources' SyncStates. A run's stage passes itself as `self`: its own order
# comes from its last run and its segments (section 6).
function waitsfor(accesses; self = nothing)
    waits = Token[]
    for (r, u) in accesses
        s = syncstate(r)
        append!(waits, eagerwaits(s, u))
        append!(waits, s.retired)
        for (pl, v) in namingplanuses(r)             # the plans naming r, and how they use it
            pl !== self && hazard(v, u) && append!(waits, lastrun(pl))
        end
    end
    return latestperchannel(filter(!passed, waits))
    # expressed by the device type: a timeline-semaphore wait (Vulkan),
    # waitForEvent:value: (Metal), stream order or a stream value wait (CUDA, HIP)
end
eagerwaits(s, u) = [a.token for a in eageraccesses(s) if hazard(use(a), u)]   # the last write and the reads since

# What is not a run records its accesses here, and tells the plans: a
# segment that reads then writes a resource is recorded as a writer.
# A run applying changes passes itself as `self`: its own inbox does not get
# them back (its later stages are ordered after its first segment anyway).
function recordeager!(accesses, token::Token; self = nothing)
    for (r, u) in accesses
        s = syncstate(r)
        a = Access(token, stages(u))
        writes(u) ? (s.lastwrite = a; empty!(s.reads)) : replaceforchannel!(s.reads, a)
        foreach(pl -> pl === self || pushinbox!(pl, r), namingplans(r))
    end
end
# `retired` keeps only the latest unpassed token per channel (pruned when a
# token is added), so it does not grow with recompiles and piece deletes.
addretired!(s::SyncState, tokens) = (s.retired = latestperchannel(filter(!passed, [s.retired; tokens])))
# Everything that may still use a resource's storage: its eager accesses,
# retired plans' last tokens, and the last runs of the plans naming it (read
# under their sync locks). Retirement of replaced or released storage waits
# for these.
# lasttokens(pl::Plan): the plan's last run (lastrun(pl)), for retiring what
# only the plan uses; lasttokens(s, r): everything that may use r's storage.
lasttokens(pl::Plan) = collect(lastrun(pl))
lasttokens(s::SyncState, r) = filter(!passed, [[a.token for a in eageraccesses(s)]; s.retired;
                                                [t for pl in namingplans(r) for t in lastrun(pl)]])

# Claim, record, submit: one critical section per channel, so submissions reach
# the queue in token order (a timeline semaphore's signal value must increase
# with every signal). The caller holds the locks, and has computed the waits
# under them. `eager` are the accesses to record on SyncStates (none for a
# run's own work: a run's accesses are its last-run tokens, set by the caller
# under its sync lock).
function submit!(ch::SubmitChannel, parts; eager = (), waits = waitsfor(eager),
                 signals = (), present = nothing, self = nothing)
    checkhealthy(channelstate(ch).domain)
    packet = preparesubmit(ch, parts, waits, signals)  # fallible, BEFORE claim
    st = channelstate(ch)
    lock(st.lock) do
        checkhealthy(st.domain)
        token = Token(ch, nexttoken!(st))
        try
            recordeager!(eager, token; self)   # pre-reserved publication storage
            submitnative!(ch, token.value, packet)
        catch err
            poison!(st.domain, err)           # native acceptance may be ambiguous
            quarantine!(st.domain, packet, eager, token) # no recycle/forged completion
            rethrow()
        end
        return token                          # accepted, not completed
    end
end
# Caller commits changes, publishes plan tokens and reader updates under its
# expanded locks. A post-accept bookkeeping failure also poisons the domain.
# `present` is prepared run metadata only; finishpresent! enqueues it after
# releasing record locks, after ALL stages were accepted. No run continuation
# is needed for an OOM present retry. A pre-WSI partial failure instead retires
# its signal semaphore and keeps/releases the acquired image for a fresh run.

# ── Locks ──
# A plan's sync lock orders its submissions against those of the plans it
# shares resources with: a stage holds its own, those of the plans sharing
# what it writes, and those of the plans naming its inbox resources (lockset);
# register! and unregister! hold those of the plans naming what the plan
# names; anything else that reads a plan's last run (an eager call on a
# resource the plan names, the retirement of storage it used) holds that
# plan's sync lock. So a last run is never read while it is about to change:
# a run sets it under its own sync lock, before releasing it. Order: plan
# sync locks by plan id, then SyncStates by id, then the channel lock.
function lockplans(f, plans)
    ps = sort(unique(plans); by = p -> p.id)
    foreach(p -> lock(p.synclock), ps)
    try f() finally foreach(p -> unlock(p.synclock), reverse(ps)) end
end
# For a resource: the sync locks of the plans naming it (a snapshot read
# first, checked under the locks), then its SyncState (and its composites'
# members', lockexpanded). Eager calls, host reads, vendor views and eviction
# copies lock this way (submitapplying).
function lockresources(f, uses; lent = false, transfer = nothing, queued = false)
    while true
        plans = namingplanssnapshot(uses)
        r = lockplans(plans) do
            lockexpanded(uses; lent) do recs, x
                namingplanssnapshot(uses) === plans || return Retry()
                waits = queued ? () : transferwaits(recs, transfer)
                isempty(waits) || return Done(WaitForTransfers(waits))
                Done(f(recs, x))
            end
        end
        done(r) && return r.value
    end
end

# SyncStates in id order, unlocked in any case. A lent one throws unless the
# caller is withview's own end or a release (section 14). `freed` is checked by
# calls made through an array's handle (eager calls, declarations, stores),
# not by runs of graphs that still hold the array
# (resizing-and-raytracing.jl B.10).
function locked(f, records; lent = false)
    rs = sort(unique(records); by = r -> r.id)
    foreach(r -> lock(r.lock), rs)
    try
        for r in rs
            lent || !r.lent || throw(ArgumentError("array lent to vendor code"))
        end
        return f()
    finally
        foreach(r -> unlock(r.lock), reverse(rs))
    end
end

# Pending changes (Change, change!, takepending, submitwith!, lockexpanded):
# resizing-and-raytracing.jl B.3-B.4. A submission that must happen now, not
# with the next use (a host read, an eviction copy, vendor views), is
# submitnow!: its own work through submitwith!, so what is pending on its
# resources is applied with it.
# Every submitter outside a run goes through this: B.4 step 1 (prepare!,
# before any lock), step 2 (submitwith!, under the locks). Again starts over;
# WaitFor waits outside the locks first. `own(after)` records the
# submitter's own commands once the steps of the pending changes are known:
# after(st) is the storage an array has after them, so an eager call packs
# the addresses its arguments will have. `own(ch, after)` gets the channel the
# submission picked under the locks, from its waits (section 8), and records
# for that channel's kind. Returns the token.
function submitapplying(own, uses; work = ComputeWork(),
                        validate = Returns(true), transfer = nothing)   # uses: (resource, use) pairs
    target = EagerQueue(device(first(first(uses))), work)
    while true
        prepared = Prepared()
        r = nothing
        try
            prepare!(prepared, uses)                  # storage, pages, capacity, command objects, restores
            r = lockresources(uses; transfer) do recs, x
                submitwith!(target, recs, prepared, expand(x, uses), OwnWork(own, validate))   # records eager accesses
            end
        finally
            release!(prepared)                        # what no committed step took
        end
        needsdestination(r) && return r  # caller must allocate a larger fixed destination
        done(r) && return first(r)
        waitforruns(r)                                # WaitFor: outside every lock; Again: nothing
    end
end
struct OwnWork{F,V}
    own::F
    validate::V
end
OwnWork(f) = OwnWork(f, Returns(true))
struct NeedDestination end
needsdestination(::NeedDestination) = true
needsdestination(_) = false
validateown(parts::Vector, after) = true
validateown(w::OwnWork, after) = w.validate(after)
waitforruns(w::WaitForTransfers) = waitfortransfers(w) # outside all record locks
recordown(parts::Vector, ch, steps) = map(p -> commandbuffer(p, steps), parts)   # a run's recorded parts
recordown(w::OwnWork, ch, steps) = w.own(ch, storageafter(steps))            # an eager call, a copy of its own
# An eager submission's queue: picked in submitwith!, under the locks, from
# the waits; a run passes the channel it already picked (section 6).
struct EagerQueue{W}
    device::Device
    work::W                                   # ComputeWork(), GraphicsWork() or CopyWork()
end
pick(e::EagerQueue, waits) = pickchannel(e.device, e.work, 0, waits)   # eager work has the default priority
pick(ch::SubmitChannel, waits) = ch
# A submission whose own work is a copy (a host read, an eviction copy): the
# same path, the copy emitted by `emit(b, after)`.
submitnow!(emit, uses; work = CopyWork(), validate = Returns(true), transfer = nothing) =
    submitapplying((ch, after) -> (b = Builder(device(ch), queuekind(ch)); emit(b, after); [finish(b)]),
                   uses; work, validate, transfer)

# ══ 9. Memory: pool, kinds, arenas, release ═══════════════════════════════════

# Never called with an sync-state, channel or pool lock held.
function acquire!(pool::Pool, kind, bytes; constraint = nothing)
    sweep!(pool)
    r = tryacquire!(pool, kind, bytes, constraint)
    r === nothing || return r
    # Pressure is handled here and only here. It submits only the unmapping of
    # idle arena pages and eviction copies, and never flushes a queue.
    # dropidlecache! drops idle native holds for pool.device (section 13 of
    # execution-contracts.md); destruction still follows retirement tokens.
    for relieve in (dropidlecache!, unmapidle!, trimblocks!, collectgarbage!)
        relieve(pool)
        sweep!(pool)
        r = tryacquire!(pool, kind, bytes, constraint)
        r === nothing || return r
    end
    while (t = oldestretiring(pool)) !== nothing      # read under the pool lock
        waitfor(t)                                    # outside it
        sweep!(pool)
        r = tryacquire!(pool, kind, bytes, constraint)
        r === nothing || return r
    end
    while true
        tokens = evictfor!(pool, bytes)
        isempty(tokens) && break
        waitfor(tokens)                               # outside every lock: the evicted storage's retirement tokens
        sweep!(pool)
        r = tryacquire!(pool, kind, bytes, constraint)
        r === nothing || return r
    end
    throw(OutOfDeviceMemory(kind, bytes, budget(pool.device)))
end

# ── Eviction and restore ──
# Arrays and images only. Acceleration structures are not evicted (a TLAS or
# BLAS would need its sources back before a rebuild).

# The contents of an evicted writable array: pageable host memory, not pool
# memory, so the OS may compress it or swap it to disk (Mantle writes no
# files). Core owns suitably aligned/importable backing for HostCopy; an
# arbitrary Vector alone does not establish that property. Bounded HostCopyJob
# below describes eviction. Restore uses prepared native access to these same
# bytes; a staged restore path requires the debug-first evidence gate (section 12).
mutable struct HostCopy
    bytes::Vector{UInt8}
    shape::Dims                     # original extent; finishhostshape! applies a copied device length
end
struct Evicted{H}
    host::H                       # a HostCopy with the contents, valid once `token` passed;
                                  #   or the host array a read-only array keeps: nothing was copied out
    token::Union{Nothing,Token}   # the eviction copy's submission; nothing for a read-only array
end

# Checked under the record's lock. Not evictable: anything a composite holds
# (a TLAS or BLAS range reads it through a cell that a restore would have
# to reach), anything a plan recorded by handle (a restore
# moves it, so the parts that recorded the handle would be recorded again),
# anything a running plan names, lent or with pending
# changes. Accesses still in flight are allowed: the eviction copy waits for
# the last write through the sync state (a read need not wait for reads),
# and the storage is retired with the tokens of every access in flight and
# the copy's. So `run!(dnn_1); run!(dnn_2)` back to back evicts dnn_1's
# weights while its run is still on the GPU, and acquire! waits for those
# tokens. A structure's own arrays (records, range table, offsets, software
# storage) are held by it with composite holds, so they are never evicted.
evictable(st) = st.composites[] == 0 && !any(d -> d.handle, values(st.dependents)) &&
                !any(running, keys(st.dependents)) && !st.sync.lent && st.sync.transfer === nothing &&
                isempty(st.sync.pending) && evictablestorage(st.storage)
evictablestorage(::Union{Region,Reserve,Committed}) = true
evictablestorage(::Union{Evicted,Released}) = false

# The last pressure step in acquire!: evicts idle resources, least recently
# used first (the token of the last access), until their bytes cover `need`,
# and returns the tokens the evicted storage was retired with, for acquire! to
# wait on. Each copy is a submission of its own; a read-only array is
# copied nowhere, its device copy is dropped. Under the record's lock the
# resource becomes Evicted and the plans that name it (all register in its
# dependents, resizing-and-raytracing.jl B.6) are marked. The same on every
# device type: the device allocation is freed, and the copy is pageable.
function evictfor!(pool, need)
    tokens = Token[]
    freed = 0
    for st in leastrecentlyused(pool)
        freed >= need && break
        snap = evictionsnapshot(st) # immutable storage/shape/bytes/source captured under expanded locks
        snap === nothing && continue
        host = evictionhost!(pool, snap, snap.source) # allocate outside locks
        job = nothing
        try
            # Readonly eviction commits directly. Writable eviction freezes the
            # physical source under expanded locks and returns a held HostCopyJob.
            job = begineviction!(st, host, snap) # eligibility + snapshot + transfer publication
            job === nothing && continue
            copyeviction!(job)            # bounded chunk copies; all waits/memcpy OUTSIDE locks
            last, reclaimed = commiteviction!(job) # same expanded locks; preserve pending requests
            append!(tokens, last)
            freed += reclaimed
        finally
            # Abort leaves old storage usable; native failure quarantines holds.
            # Complete/abort wakes transfer waiters; no graph lock is needed.
            job === nothing ? release!(pool, host) : finisheviction!(job)
        end
    end
    return tokens
end
# Pageable copy job, synchronous only inside pressure. Native submissions read
# the frozen source directly: NEVER apply pending stores/resizes in this job.
function copyeviction!(job::HostCopyJob)
    for chunk in chunks(job.snapshot, job.staging; limit = 8 << 20)
        ch = pickchannel(job.device, CopyWork(), 0, job.writers)   # section 8
        b = Builder(job.device, queuekind(ch))
        emitcopy!(b, nothing, copyof(chunk.source => job.staging), chunk.deps)
        native = finish(b)
        token = lockresources(((job.state, Read()),); transfer = job) do recs, x
            matchestransfer(job) || throw(ArgumentError("eviction snapshot lost"))
            t = submit!(ch, [native]; eager = ((job.state, Read()),), waits = job.writers)
            remember!(job, t) # pre-reserved token/command bookkeeping
            t
        end
        waitfor(token)                     # no Mantle record/channel/pool/window lock
        invalidatehost!(job.device, job.staging, chunk.stagingrange)
        copyto!(job.host.bytes, chunk.hostrange, hostview(job.staging), chunk.stagingrange)
    end
    finishhostshape!(job) # copied length cell if device-written; validate <= copied capacity
end
# commiteviction! publishes Evicted(host,nothing), old-use retirement, restore
# marks and transfer completion together. Readonly jobs have no GPU copy.
# Complete protocol and staging-lease order: execution-contracts.md §12.
evictionhost!(pool, snap, ::Nothing) = HostCopy(pagealigned(UInt8, snap.bytes), snap.shape)
evictionhost!(pool, snap, source::AbstractArray) = source
release!(pool, ::HostCopy) = nothing                # Julia memory: the GC frees it once the Evicted is gone
release!(pool, ::AbstractArray) = nothing           # the caller's host array is never Mantle's to release

# Restore: one of the steps a submitter applies (resizing-and-raytracing.jl
# B.4), the same shape as any other. Before its locks, prepare! allocates
# restore storage, of the requested size, for each evicted resource the
# submission will lock: a run does that only when its plan is marked (an
# eviction marks the plans in its dependents, and a new plan starts marked),
# so an unmarked run looks for nothing; an eager call looks at its arguments.
# This allocation's own pressure step cannot evict what a running plan names
# (pl.running is set first). Under the locks submitwith! emits, for each
# evicted resource locked, a Restore step and a cell write with the prepared
# storage, in front of the changes pending on it: the Restore writes the
# resource, so it waits for the eviction copy (recorded as a read). Both
# sources are retained host memory: a HostCopy or the immutable source a
# read-only array keeps. Before record locks, prepare! either acquires a native
# source view/registration or initializes the private CPU-visible final storage
# directly and finishes host visibility/flush. It creates no intermediate upload
# allocation. RestoreStorage retains the destination and any native source lease.
# On retry/preclaim failure, release unused preparation and retain Evicted state;
# accepted native copies retain their source view until their copy token passes.
# Both preparations remain one logical resource write at acceptance/publication.
# A direct-path failure must be debugged before proposing any design change;
# staged restore is not an assumed alternative (F74, execution-contracts.md 12).
# Storage missing/too small after locking: Again, nothing emitted. No eviction
# can happen while these locks are held.
restoresteps(st, ::Union{Region,Reserve,Committed}, _) = ()   # not evicted, or restored meanwhile
restoresteps(st, ::Evicted, ::Missing) = Again()
# restorecopies(st,new,n): core's prepared GPU copy steps over the held source
# view, or () if prepare! already initialized the unpublished final allocation
# on the CPU. Snapshot/capacity validation is required on both paths (section 12).
restoresteps(st, ev::Evicted, new::RestoreStorage) = sizeof(new.storage) >= bytes(st) ?
    (restorecopies(st, new, min(sizeof(ev.host), bytes(st)))...,
     CellWrite(st, address(new.storage), st.dims)) : Again()
Base.sizeof(h::HostCopy) = sizeof(h.bytes)
isevicted(::Evicted) = true
isevicted(_) = false
emitstep!(b, s::Restore) = emitcopy!(b, nothing, copyof(s.source => s.new, s.bytes), Deps((), ()))

# Everything else that meets an evicted array, by dispatch on the storage:
# - an eager call, a store, a resize!, becoming a composite's member: the
#   submission that applies them restores it first (above); the calls only
#   enqueue;
# - a pending resize!: the restore allocates the requested size and copies
#   min(host copy, requested) bytes, so the Resize after it is in place;
# - a host read: waits for the eviction token and copies from the host copy
#   under the record's lock, no submission (a read-only array: from the host
#   array it keeps, no wait);
# - an image: restored into a new image under a new, unused table index, like
#   a regrown texture (resizing-and-raytracing.jl B.9): commands submitted
#   before the eviction may still read the old index;
# - the last hold: the host copy is dropped after the eviction token (section
#   10; Julia memory, the GC frees it); a read-only array's host array is the
#   caller's, nothing is released.
# The pool's least-recently-used list holds weak references, so it keeps
# nothing alive; a released state leaves it (section 10).

# The budget check and the new block happen under one lock, so two threads
# cannot both pass it. rawalloc is a driver call, not a GPU wait.
tryacquire!(pool, kind, bytes, constraint) = lock(pool.lock) do
    class = allocationclass(pool, kind, constraint)   # core chooses from immutable memory-class facts
    c = withclass(constraint, class)
    r = suballocate!(pool, kind, bytes, c)
    r === nothing || return r
    # budget is ADDITIONAL available bytes; never add used(pool) again.
    blocksize(bytes) > budget(pool.device, class.heap) && return nothing
    addblock!(pool, rawalloc(pool.device, kind, blocksize(bytes), c))
    suballocate!(pool, kind, bytes, c)
end
# Vulkan: heapBudget-heapUsage; CUDA/HIP: free bytes; Metal: recommended headroom
# estimate. The native normalization and quality are execution-contracts.md §12.

# Retired with the tokens of its last uses; released once all have passed.
# Every release!/raw-free path also checks pool.domain: failed-domain objects
# are quarantined even when the caller has no token after an enqueue throw.
retire!(pool, x, tokens) = push!(pool.inbox, Retired(x, tokens))   # lock-free: safe in a finalizer

function sweep!(pool)
    healthy(pool.domain) || return quarantineall!(pool) # includes tokenless cleanup after native failure
    for x in drain!(pool.inbox)                      # lock-free
        settle!(pool, x)     # DropCreator, Release and finalizers' hold drops do their work here, outside
    end                      #   the pool lock; a Retired or a DropAfter joins pool.retiring
    done = lock(pool.lock) do                        # a leaf: only the list changes inside
        d = filter(x -> all(passed, x.tokens), pool.retiring)
        setdiff!(pool.retiring, d)
        d
    end
    foreach(x -> release!(pool, x), done)            # regions back to their blocks, driver objects destroyed,
end                                                  #   a DropAfter's hold dropped (section 10)

# A sparse arena: a fixed virtual range; pages mapped as tenants need them. Its
# state (mapped, tenants) is protected by its sync state's lock; binding and
# unbinding go to one channel, the device's sparse-binding queue.
mutable struct Arena                # one arena part
    range::VirtualRange             # never moves
    mapped::Int
    tenants::Dict{Int,Int}          # graph id => its peak; the arena needs the largest current one
    const sync::SyncState
    holder::Union{Nothing,Task}     # the run holding it until its last stage is submitted (section 6)
    const cond::Threads.Condition   # protects holder; waited on by the next run
end

function unmapidle!(pool)           # a pressure step; acquire! holds no record lock
    for a in pool.arenas
        locked((a.sync,); lent = true) do
            any(id -> running(id), keys(a.tenants)) && return   # a run in progress keeps its pages
            need = pageceil(maximum((peak for (id, peak) in a.tenants if !idle(id)); init = 0))
            need < a.mapped || return
            token = sparsesubmit!(sparsechannel(a), unbindpages!, unbinds(a, need:a.mapped), ((a, Write()),))   # after every recorded use
            retire!(pool, pagesof(a, need:a.mapped), (token,))  # the pages go back once unbound
            a.mapped = need
            foreach(id -> markunmapped!(id, a), (id for (id, peak) in a.tenants if peak > need))
        end
    end
end

# QueueMapping is a real queue operation (vkQueueBindSparse or a supported
# Metal buffer mapping update), not CUDA/HIP host MemMap or a recorded command. Core claims a token on the
# sparse channel and records the access under the channel lock; the device
# type's verb binds or unbinds and signals the token. Callers hold the
# records' locks.
function sparsesubmit!(ch, verb!, ranges, accesses)
    waits = waitsfor(accesses)
    lock(channelstate(ch).lock) do
        token = Token(ch, nexttoken!(channelstate(ch)))
        recordeager!(accesses, token)            # binding and unbinding are not runs: recorded on the SyncState
        try
            verb!(ch, token.value, ranges, waits) # pre-prepared bind/unbind packet
        catch err
            poison!(channelstate(ch).domain, err)
            quarantine!(channelstate(ch).domain, ranges, accesses, token)
            rethrow()
        end
        token
    end
end
# markunmapped!(id, a): marks part `a` of the plan found by id (never kept alive): its
# pl.unmapped[a], an atomic flag per arena part, set; clearunmapped!(id, a) clears it.
# A new plan starts marked, so its first run maps what it needs. A run sets
# `running` before it looks at the mark, so an unmap either happens before it
# (the run sees the mark and binds) or waits until the run has ended. Pressure
# never unmaps pages of an array beyond its size: views may read them
# (resizing-and-raytracing.jl B.2); trim!(a) does that on request.

# A plan has several arena parts (arenaparts(pl): the device-memory part, the
# host-visible part host nodes use, the part of window-sized transients), each
# with the plan's peak in it (peak(pl, a); for the window part, the pages the
# current generation's size needs). Every step below goes over all of them.

# In run!: pages for the part of each range not mapped now (a.mapped read
# without the lock; bindunmapped! checks under it), before any lock.
function acquireunmapped!(rs, pl)
    for a in arenaparts(pl)
        pl.unmapped[a][] || continue                 # only the parts a mark names
        n = pageceil(peak(pl, a)) - a.mapped
        n > 0 && (rs.pages[a] = acquirepages!(pool(pl.device), a, n))   # one part at a time: a throw leaves rs whole
    end
end

# In run!, under the arena parts' record locks (stage 1 locks them). The marks
# tell the run which parts to bind; nothing at run decides whether to. The
# one comparison checks that pages allocated without the lock still cover the
# gap, which another tenant's pressure step may have widened in between: the
# retry check of an optimistic allocation, like lockexpanded's snapshot
# check, not a detection of a change. Again, and the next attempt allocates
# again. The pages it binds leave rs; the rest stay there for run!'s finally.
# A tenant's mark on a part is cleared only for that part, and only once
# that part covers the tenant's peak in it.
function bindunmapped!(pl, rs)
    marked = [a for a in arenaparts(pl) if pl.unmapped[a][]]
    isempty(marked) && return Mapped()
    all(a -> covers(get(rs.pages, a, nothing), a.mapped:pageceil(peak(pl, a))), marked) || return Again()
    for a in marked
        lo, hi = a.mapped, pageceil(peak(pl, a))
        if hi > lo
            sparsesubmit!(sparsechannel(a), bindpages!, binds(a, rs.pages[a], lo:hi), ((a, Write()),))
            rs.pages[a] = pagesoutside(rs.pages[a], lo:hi)
            a.mapped = hi
        end
        foreach(id -> clearunmapped!(id, a), (id for (id, pk) in a.tenants if pk <= a.mapped))
    end
    return Mapped()
end
# The window part's peak follows the swapchain generation: recordwindows!
# (graph lock held) sets the plan's tenancy there to the generation's size
# under the part's lock and, when it grew, marks that part for the plan, so
# the next run binds what the larger images need; unmapidle! sizes from the
# tenancies, so it sees the new peak too.
# A part without sparse residency (the host-visible one; any part on such a
# device) is mapped whole when it is created and never unmapped: covers is
# true for it and nothing is bound.

# ══ 10. Holds on shared resources ═════════════════════════════════════════════

# A hold never goes from zero back to one.
function addhold!(st)
    while true
        n = st.holds[]
        n == 0 && throw(ArgumentError("use of a released resource"))
        Threads.atomic_cas!(st.holds, n, n + 1) == n && return
    end
end
drophold!(st) = Threads.atomic_sub!(st.holds, 1) == 1 && release!(st)
# A composite's hold (a TLAS or BLAS range) is counted twice:
# in holds, and in composites, which keeps the array from being evicted
# (section 9).
# Taken under the array's record lock (holdmember!, resizing-and-raytracing.jl
# B.8), so evictable, checked under the same lock, sees both or neither.
addcompositehold!(st::ArrayState) = (addhold!(st); Threads.atomic_add!(st.composites, 1))
dropcompositehold!(st::ArrayState) = (Threads.atomic_sub!(st.composites, 1); drophold!(st))
addcompositehold!(x) = addhold!(x)      # a BLAS held by a range: never evicted anyway
dropcompositehold!(x) = drophold!(x)
# A hold dropped once tokens have passed: a composite's old member after the
# submission that applied its replacement (resizing-and-raytracing.jl B.4,
# deferred drops), a vendor view's hold after the stream's work. Through the
# inbox, so it is safe anywhere; sweep! drops it once every token passed.
dropafter!(st, tokens, drop = drophold!) = push!(pool(st).inbox, DropAfter(st, tokens, drop))

# The last hold, once. Under the record's lock: the pending changes are
# dropped (staging retired with no tokens, storage a change prepared released,
# deferred hold drops done), the storage becomes Released() so a racing
# eviction or store sees it, the old storage (an Evicted's host copy included)
# is retired with the record's last tokens, the state leaves the pool's
# eviction list, and its cell goes back to the cell arena with the same tokens.
function release!(st::ArrayState)
    locked((st.sync,); lent = true) do
        tokens = lasttokens(st.sync, st)
        dropall!(st.sync.pending); empty!(st.sync.pending)
        foreach(d -> d(tokens), st.sync.deferred); empty!(st.sync.deferred)
        old, st.storage = st.storage, Released()
        retire!(pool(st), old, tokens)
        retire!(pool(st), st.cell, tokens)
        forget!(pool(st), st)                     # leaves the least-recently-used list
    end
end

# At declaration (holds keep it alive, so a plain read is enough); calls that
# claim a token check under the record lock instead (`locked`).
usable(a::MantleArray) = usable(a.state)
usable(st::ArrayState) = st.sync.freed && throw(ArgumentError("use of an array after free!"))
usable(r::GPURef) = r.sync.freed && throw(ArgumentError("use of a GPURef after free!"))

# Every resource a node names (heldresources: all but the graph's
# transients), wherever it sits in an argument (inside a Broadcasted, a
# keyword, a view's parent, a when! flag, a repeated group's count): one hold per
# graph, counted per node so delete! knows when the last node naming it went.
function name!(g::Graph, n::Node)
    for st in unique(heldresources(n))
        k = get(g.named, st, 0)
        k == 0 && addhold!(st)
        g.named[st] = k + 1
    end
end

# free! of a root array drops the creator's hold once, whatever the number of
# threads; the state's finalizer does the same through the inbox. A view is
# the same type, told apart by its own dims.
function free!(a::MantleArray)
    a.view === nothing || throw(ArgumentError("free! of a view; free its parent"))
    dropcreator!(a.state)
end
dropcreator!(st) = locked(() -> st.sync.freed ? false : (st.sync.freed = true), (st.sync,); lent = true) &&
                   drophold!(st)

# free!(g) and the graph's finalizer: one guard, one function. The swap is
# inside lockgraph, so free!(g) from g's own host function throws before
# anything changed (a swap first would leave `freed` set and nothing released,
# since the finalizer then skips release!).
free!(g::Graph) = lockgraph(g) do
    (@atomicswap g.freed = true) || release!(g)
end
function release!(g::Graph)
    foreach(drophold!, keys(g.named)); empty!(g.named)   # g.plan stays: declarable and run! check g.freed
    foreach(drophold!, g.unnamed); empty!(g.unnamed)     #   (the plan still names these: retired below with its tokens)
    g.plan === nothing && return
    pl = g.plan
    locked(() -> foreach(a -> untenant!(a, pl.graphid), arenaparts(pl)), [a.sync for a in arenaparts(pl)]; lent = true)
    unregister!(g.plan)                              # off the dependents and namers
    retire!(pool(g.device), g.plan, lasttokens(g.plan))   # argument block, recordings, descriptors
end
# Finalizers only push onto the lock-free inbox; settle! (in sweep!) does the
# work, one item at a time:
#   finalizer(st -> push!(pool(st).inbox, DropCreator(st)), state)   # settle!: dropcreator!(st)
#   finalizer(g -> push!(pool(g.device).inbox, Release(g)), g)       # settle!: (@atomicswap g.freed = true) || release!(g)
#   finalizer(x -> push!(pool(x).inbox, DropHolds(x)), x)            # a TLAS, BLAS or RenderObject:
#                                                                    #   settle! drops the composite holds it has
# The state's finalizer runs only when no array, view, graph or composite
# refers to it (g.named and the composites' members keep it alive), so it
# drops the creator's hold if free! did not; other holders drop theirs through
# their own finalizers. A finalized graph cannot be running (run! holds a
# reference), so its release needs no graph lock. `settle!` takes records
# without the lent check and never waits on a run.

# ══ 11. Eager calls: the same dispatch!, submitted now ════════════════════════

# A GPURef argument is packed with its slot's address; a pending store
# into the slot is applied in front of the launch by the same submission.
# Conditions exist in graphs only (when! takes a Graph or a Group): an
# eager call runs. Its arguments' pending changes, and the restore of an
# evicted one, are applied in front of the launch by the same submission
# (submitapplying). A write to a read-only array throws first, as in a graph
# (checkwritable, section 2).
# As in a graph, the launch runs whole: an estimate past the job timeout throws
# LaunchTooLong before anything is acquired, unless `split = true`, which
# submits its workgroup ranges (costs!, section 4) as consecutive calls.
function dispatch!(dev::Device, kernel, args::Tuple, ndrange; origin = nothing, split = false)
    checkwritable(kernel, args)                           # before anything is compiled or acquired
    k = kernel!(dev, kernel, argtypes(args))
    packed = acquire!(pool(dev), Unified(), argumentbytes(k))   # fallible: before any lock
    token = nothing
    try
        token = submitapplying(uses(k, args); work = ComputeWork()) do ch, after   # under the locks, after the
                                                                                   #   pending changes' steps
            foreach(usable, resources(args))         # freed through the creator's handle: throws, under the lock
            packarguments!(packed, k, args, after)   # each array's storage after the pending changes, its requested dims
            ext = launchextents(dev, k, resolve(ndrange, args))   # requested lengths; device lengths use indirect / capacity predication
            b = Builder(dev, queuekind(ch))         # ch: picked under the locks, from the waits (section 8)
            emitdispatch!(b, nothing, CompiledDispatch(k, ext, packed), Deps((), ()))
            [finish(b)]
        end
    finally
        retire!(pool(dev), packed, token === nothing ? () : (token,))   # after the call ran, or now after a throw
    end
    return nothing
end

# The temporary form, the same as on a graph: on a device a region retired with
# its last uses' tokens, on a graph a transient.
MantleArray(f, dev::Device, T, dims...) = (t = MantleArray(dev, T, dims...); f(t); free!(t))
MantleArray(f, l::GraphDevice, T, dims...) = f(MantleArray(l, T, dims...))

# A host read: a submission of its own (the pending changes on the array, a
# copy into Readback memory), a wait for its token, a memcpy. For an array whose length a kernel writes, the cell is copied in the
# same submission and the data up to the capacity, so length and data come from
# the same point in the queue.
Base.Array(a::MantleArray{T,N,<:Device}) where {T,N} = Array(a, memorykind(a))
# An array in unified memory (`unified = true`, a graph's output): its pending changes first (a
# submission of their own only if there are any), then a wait for every write
# to it, eager or by a run (waitsfor: the runs' last tokens are read under the
# naming plans' sync locks; the wait happens outside the locks), and a memcpy.
function Base.Array(a::MantleArray{T,N}, ::Unified) where {T,N}
    uses = ((a.state, Read()),)
    while true
        usable(a)
        pending = lockresources(uses) do recs, x
            usable(a)
            !isempty(a.state.sync.pending)
        end
        pending && submitnow!((b, after) -> nothing, uses)
        waits, snap = lockresources(uses) do recs, x
            usable(a)
            isempty(a.state.sync.pending) || return (Token[], nothing)
            w = waitsfor(uses)
            all(passed, w) ? (Token[], hostreadsnapshot(a)) : (w, nothing)
        end
        if snap === nothing
            waitfor(waits)                       # outside every record/plan/channel lock
            continue
        end
        out = Array{T,N}(undef, snap.dims)           # host allocation outside the locks
        result = lockresources(uses) do recs, x
            usable(a)
            matcheshostread(a, snap) || return Again()
            copyto!(out, hostview(snap))             # CPU memcpy only, no GPU wait or allocation
            out
        end
        again(result) || return result
    end
end
# hostreadsnapshot captures storage, applied dims, view mapping, pending-change
# version and every naming writer's tokens while they are passed and locked.
# matcheshostread rechecks them under the same expanded locks. Every writer
# submits under those locks, so none can start during the CPU copy. A wait
# followed by an unlocked memcpy would race the next run/move/free.
# For a kernel-written length, snapshot dims come from the now-idle mapped
# cell, under the locks. This direct path serializes submissions for the
# duration of its CPU copy; callers can use staged output when that cost
# outweighs avoiding a GPU copy.
function Base.Array(a::MantleArray{T,N}, _) where {T,N}
    uses = ((a.state, Read()),)
    while true
        bytes = lockresources(uses) do recs, x
            usable(a)
            projectedreadbackbytes(a)   # immutable pending snapshot; not size() readback
        end
        staging = acquire!(pool(a), Readback(), bytes + cellbytes(a))
        shape = Ref{Union{Dims{N},DeviceLength}}()
        token = nothing
        try
            r = submitnow!(uses;
                validate = after -> readbackbytes(a, after(a.state)) <= bytes) do b, after
                # validate runs against the post-change state BEFORE bind/claim;
                # a raced growth returns NeedDestination and retires unused staging, then retries.
                nbytes = readbackbytes(a, after(a.state))
                shape[] = copyshape(a, after(a.state))
                emitcopy!(b, nothing, copyof(region(after(a.state), a) => staging, nbytes), Deps((), ()))
                foreach(c -> emitcopy!(b, nothing, c, Deps((), ())), cellcopies(a, staging))
            end
            needsdestination(r) && continue
            token = r
            waitfor(token)
            dims = readdims(staging, shape[]) # validate copied device length <= copied capacity
            return copyto!(Array{T,N}(undef, dims), hostview(staging, T, dims))
        finally
            retire!(pool(a), staging, token === nothing ? () : (token,))
        end
    end
end
# An evicted array: Array(a, kind, ::Evicted) waits for the eviction token and
# copies from the host copy, holding the record's lock (section 9).

# A length a kernel writes, for size(a) and length(a) (section 1; Julia's
# array API, so no second name): a submission of its own that
# applies what is pending on the array and copies its cell into Readback
# memory, ordered after every write to the array, eager or by a run
# (submitnow! records a read: its waits are the last write and the last runs
# of the plans writing it), then a wait for its token and a memcpy.
function readlength(st::ArrayState)
    usable(st)
    staging = acquire!(pool(st), Readback(), cellbytes(st))
    token = nothing
    try
        token = submitnow!(((st, Read()),)) do b, after
            foreach(c -> emitcopy!(b, nothing, c, Deps((), ())), cellcopies(st, staging))
        end
        waitfor(token)
        return readdims(st, staging)
    finally
        retire!(pool(st), staging, token === nothing ? () : (token,))
    end
end

# ══ 12. Device-type verbs (examples of the only code a device type writes) ════

# Vulkan: the barrier core derived, then the launch. The one barrier call site.
# A when! launch is indirect like any other; the flag was multiplied into its
# command by the head or a prepare node (section 2).
function emitdispatch!(b::VulkanBuilder, p, cmd::CompiledDispatch, deps::Deps)
    emitbarrier!(b, deps.barrier)
    vkCmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cmd.kernel.native)
    vkCmdPushConstants(b.cmd, layout(cmd), argumentaddress(cmd))   # the block's address, or the packed region's
    emitlaunch!(b, cmd.extents)                      # vkCmdDispatch, or vkCmdDispatchIndirect from its command slot
end
# A small store in a change prefix (resizing-and-raytracing.jl B.3): the bytes
# go into the command buffer (vkCmdUpdateBuffer, at most 65536 bytes). The
# barrier around it is emitstep!'s.
emitstore!(b::VulkanBuilder, dst, bytes) = vkCmdUpdateBuffer(b.cmd, buffer(dst), offset(dst), length(bytes), bytes)

# CUDA: the edges core derived. Handles are kept per node; a predecessor in
# another segment is a segment wait (crossqueuewaits!), not an edge.
function emitdispatch!(b::CudaBuilder, p, cmd::CompiledDispatch, deps::Deps)
    preds = [b.handles[q] for q in deps.preds if haskey(b.handles, q)]
    b.handles[p] = cuGraphAddKernelNode(b.graph, preds, kernelparams(cmd))
end

# The prepared packet already owns native arrays; no submitinfo allocation
# after claim. Its acquire wait and rendered signal are ordinary native fields.
function submitnative!(ch::VulkanChannel, value, packet)
    setpreparedsignal!(packet, value)
    checkresult(vkQueueSubmit2(ch.queue, packet.native))
end
# Separate WSI operation, outside plan/resource locks. Native queue external
# synchronization still requires the canonical ChannelState lock.
function presentnative!(ch::VulkanChannel, request)
    result = vkQueuePresentKHR(ch.queue, request.info)
    presentoutcome(result, request) # distinguish not-enqueued OOM from enqueued OUT_OF_DATE
end
passed(ch::VulkanChannel, value) = vkGetSemaphoreCounterValue(ch.timeline) >= value

rawalloc(dev::LavaDevice, kind, bytes, constraint) =
    vkAllocateMemory(dev.handle, allocateinfo(memorytype(dev, kind, constraint), bytes))

# ══ 13. Devices ═══════════════════════════════════════════════════════════════

# The one process-level state: loaded device types, opened devices (one object
# per hardware device and driver), and the Vulkan instance, created at the first
# Device() or devices() (so Lavapipe_jll is loaded before that).
struct DeviceRegistry
    lock::ReentrantLock
    types::Vector{Type}             # LavaDevice (MetalDevice on macOS); CudaDevice/ROCmDevice from extensions
    opened::Dict{HardwareKey,Device}   # the device type and the driver's device UUID => the device object
    instance::Base.RefValue{Union{Nothing,VkInstance}}
end
const REGISTRY = DeviceRegistry(ReentrantLock(), Type[], Dict{HardwareKey,Device}(), Ref{Union{Nothing,VkInstance}}(nothing))
# livegraphs(REGISTRY): the graphs of every opened device, held weakly by the
# device (for inhostfunction, resizing-and-raytracing.jl B.3).
# Mantle registers the platform native type (MetalDevice on macOS, otherwise
# LavaDevice); package extensions register CudaDevice and ROCmDevice.

devices() = lock(REGISTRY.lock) do
    [get!(() -> T(key), REGISTRY.opened, key) for T in REGISTRY.types for key in hardwarekeys(T, REGISTRY)]
end
Device() = defaultdevice(devices())   # LavaDevice on GPU 0; MetalDevice on macOS
# A device's driver objects are created on its first use, not by devices().

# ══ 14. Vendor views: scoped, with an explicit stream ═════════════════════════

# The stream is a channel like Mantle's own: a counter it writes
# (cuStreamWriteValue64) is its token. While lent, every Mantle call that locks
# the array's record throws: eager calls, stores, resize!, runs. The block
# holds the array (a free! from another thread cannot release memory the
# stream uses; the hold drops once the stream's work has passed). Restoring,
# applying the pending changes and lending happen in one locked section, so
# no change, move or eviction lands between them (eviction skips lent
# records): the vendor code sees the array's current contents at a bound,
# current address. Caller exclusively owns the foreign stream during this scope;
# signal/enqueue failures poison the domain and quarantine the lease hold, never
# drop it with an empty token set. The same rule covers errors in the callback.
function withview(f, ::Type{CuArray}, a::MantleArray{T,N,CudaDevice}, stream) where {T,N}   # scoped vendor lease
    st = a.state
    ch = foreignchannel(device(a), stream)                          # before the hold: nothing to undo if it throws
    addhold!(st)                                         # throws for a released array
    token = nothing
    try
        v = nothing
        while v === nothing                              # the same two steps as every submitter (B.4)
            prepared = Prepared()
            r = nothing
            try
                prepare!(prepared, ((st, Write()),))     # storage the pending changes and a restore need
                r = lockresources(((st, Write()),)) do recs, x   # the naming plans' sync locks, then st's
                    usable(a)                            # freed through the creator's handle: throws, under the lock
                    s = needsapplying(st) ? submitwith!(EagerQueue(device(a), CopyWork()), recs, prepared, (), OwnWork((ch, after) -> [])) :
                                            nothing      # nothing pending, not evicted: no submission
                    s === nothing || done(s) || return s # Again or WaitFor: nothing submitted
                    ready = s === nothing ? Token[] : [first(s)]
                    streamwaits!(ch, [ready; waitsfor(((st, Write()),))])   # the stream waits for Mantle's accesses
                    w = vendorview(device(a), st.storage, T, applieddims(st), strides(a))   # applied storage
                    st.sync.lent = true                # nothing after it can throw
                    notifyplans!(st)                   # lock-free: a plan naming it locks it at its next stage
                    Lent(w)                            #   and finds it lent (locked throws), as the contract says
                end
            finally
                release!(prepared)                       # what no committed step took
            end
            done(r) ? (v = r.view) : waitforruns(r)      # WaitFor: outside every lock; Again: prepare again
        end
        try
            GC.@preserve a f(v)     # no Mantle lock held while vendor code runs
        finally
            token = locked((st.sync,); lent = true) do # claim, record and signal in one step, in order
                t = lock(channelstate(ch).lock) do
                    t = Token(ch, nexttoken!(channelstate(ch)))
                    recordeager!(((st, Write()),), t)    # Mantle's later uses wait for the block's work; tells the plans
                    signal!(ch, t.value)             # written by the stream after the block's work
                    t
                end
                st.sync.lent = false
                t
            end
        end
    finally
        healthy(channelstate(ch).domain) ?
            (token === nothing ? drophold!(st) : dropafter!(st, (token,))) :
            quarantinehold!(channelstate(ch).domain, st)
    end
end
