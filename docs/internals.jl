# How the internals in `recording-plan.md` work, as Julia-flavoured pseudocode.
#
# Not runnable. It shows which function owns which decision, what it reads and
# what it writes, and which lock protects what. Helper functions that are not
# the point are left undefined. "(open)" marks names or details the plan has
# not decided.
#
# Sections follow the life of a graph: types, declaration, selection,
# compile, record, run, changes, ordering, memory, holds, eager calls,
# device-type verbs, devices, vendor views.
#
# Locks, in the only order they are taken (plan, "Ordering across graphs and
# queues"): graph lock → the run's hold on its arena parts by id (section 6)
# → window lock → plans' sync locks by plan id (section 8) → sync states by
# id → channel lock.
# Leaves, nothing else taken inside them and no GPU wait inside them: the pool
# lock, the inbox (lock-free), a value GPURef's lock (section 7), the texture
# and sampler tables' locks, a plan's runend lock (endbetween!, taken under
# record locks).
# While the host waits (for a previous run, a shared arena part, a host
# node's inputs, the swapchain acquire, a pressure step's tokens) no
# sync-state, channel,
# pool or window lock is held; the graph lock and the run's arena parts may
# be (section 6). A change waiting for a run to end (stalewait,
# resizing-and-raytracing.jl B.3) holds no lock at all and is never inside a
# run. No Mantle lock is held while vendor code runs (section 14).

# ══ 1. Types ══════════════════════════════════════════════════════════════════

abstract type Device <: KA.Backend end      # LavaDevice, MetalDevice, CudaDevice, ROCmDevice

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
    named::IdDict{Any,Int}          # resources it holds, with the number of declarations naming each (section 10)
    unnamed::Vector{Any}            # resources no node names any more; their holds drop after the plan naming them retires
    block::Union{Nothing,NodeBlock} # the insert! block being declared, or nothing
    declarer::Union{Nothing,Task}   # the task declaring under the graph lock (nested declarations reuse it)
    rewrites::Tuple                 # opted-in rewriters, empty by default
    plan::Union{Nothing,Plan}       # compiled state; a declaration or delete! marks it for recompile, the graph stays
    pending::IdDict                 # hostwritten data written while no current plan exists
    @atomic recompile::Bool         # set by markstale! (resizing-and-raytracing.jl B.6), a graph array or window
                                    # beyond its placed size, a failed compile, nodes declared or deleted after compile
    @atomic windowchanged::Bool     # set by resize!(window)
    @atomic freed::Bool
    inbody::Bool                    # declaring a repeat! body or a when! block (declarations hold the graph lock)
    inloop::Bool                    # declaring a repeat! body: its nodes are named with their LoopNode
    condition::Any                  # the flag of the enclosing when! block, or nothing
end

# Graph arrays point at this, never at the Graph: a graph is not a device.
struct GraphDevice
    graph::WeakRef
end

# What an array and all its views share: ArrayState, with its storage, host
# view, cell, sync state, holds, freed mark, dependents and readers, is
# defined in resizing-and-raytracing.jl B.2. The finalizer is on that object
# (section 10).

# One array type for every location D (a device type or GraphDevice).
struct MantleArray{T,N,D,R} <: AbstractGPUArray{T,N}
    location::D
    state::ArrayState
    role::R                         # Root(), or Derived(offset, dims, strides) for a view
end
# A view, reshape or transpose (GPUArrays' `derive`) shares its parent's state:
# a store through a view is ordered like one into the parent, and a freed
# parent makes its views unusable. A view has no hold of its own; free! and
# resize! on a view throw. Its slot entries carry its static offset and its
# own dims and strides. Shrinking never unmaps an array's pages, so a view
# never reads unmapped memory (resizing-and-raytracing.jl B.2).
Base.size(a::MantleArray) = size(a, a.role)
Base.size(a, ::Root) = a.state.dims
Base.size(a, d::Derived) = d.dims

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
    queue::Queue                    # schedule!: a channel, or HostSide() for a host node
    preds::Vector{Node}             # dependencies!: direct predecessors only
    barrier::Barrier                # barriers!, image transitions included
    command::Any                    # compile!: CompiledDispatch / CompiledTrace / CompiledCopy / …
    cost::Float64                   # costs!
    slots::UnitRange{Int}           # bindings!: its slice of the plan's argument block
end

# Every declared node keeps its origin: the Julia call that launched it (for rewriters).
struct Origin
    f::Any; args::Tuple
end

mutable struct KernelNode{K,A,R,W} <: Node  # dispatch!(location, kernel, args, ndrange)
    const kernel::K; const args::A; const ndrange::R
    const when::W                           # nothing, or a when! flag: launched indirect, count × flag (section 2)
    const origin::Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct TraceNode{P,A,R,W} <: Node   # trace!(g, pipeline, accel, args, ndrange)
    const pipeline::P; const args::A; const ndrange::R; const when::W
    const origin::Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct LoopNode{F} <: Node          # repeat!: launches only, declared once
    const count::Int; const flag::F
    const body::Vector{Node}                # KernelNodes, TraceNodes, PrepareNodes
    const origin::Origin
    compiled::Union{Nothing,CompiledNode}
end
mutable struct HostNode{F,A} <: Node        # HostCall(f); writes declared
    const f::F; const args::A; const writes::Tuple
    const origin::Origin
    compiled::Union{Nothing,CompiledNode}
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
# CopyNode (images, host transfers), RenderNode (targets and draws), BuildNode
# (a TLAS or BLAS build or refit), ForeignNode (a closed library), RegionNode
# (a rewriter's merged run), VideoDecodeNode and PresentNode have the same
# shape: data, origin, `compiled`.

# What a plan holds of one array or GPURef in its argument block: the address,
# the size, the KA ndrange values and launch commands over it, a GPURef's
# value. A resource of the plan like any other, so the head and prepare nodes
# that write it and the launches that read it get edges and barriers from the
# same rules as data.
struct Bound{R}
    resource::R
end

# The host side of a shared resource (a device array, an image, an
# acceleration structure, an array GPURef, an arena part). Runs are not
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
    deferred::Vector{Any}           # hold drops due once the submission applying `pending` is made
                                    #   (resizing-and-raytracing.jl B.4); not changes, so coalescing never drops them
    lent::Bool                      # inside a withview block (section 14)
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
location(r::GPURef) = r.device      # both forms live on a device (resizing-and-raytracing.jl B.8b)
join(a::Device, b::Device) = a === b ? a : throw(ArgumentError("arguments from two devices"))
join(g::GraphDevice, d::Device) = device(g) === d ? g : throw(ArgumentError("arguments from two devices"))
join(d::Device, g::GraphDevice) = join(g, d)
join(a::GraphDevice, b::GraphDevice) = a === b ? a : throw(ArgumentError("arguments from two graphs"))

# Everything that touches data throws on a graph array; no tracing.
Base.getindex(::MantleArray{T,N,GraphDevice}, i...) where {T,N} = nodata()
KA.get_backend(::MantleArray{T,N,GraphDevice}) where {T,N} = nodata()

# The one launch verb. On a graph device it adds a node; on a device it runs now
# (section 11). An operation's method calls it and does not know which.
function dispatch!(l::GraphDevice, kernel, args::Tuple, ndrange; origin = Origin(kernel, args))
    g = graph(l)
    declaring(g) do
        foreach(usable, resources(args))            # first: a throw leaves the graph unchanged
        addnode!(g, KernelNode(kernel, args, ndrange, g.condition, origin, nothing))   # tagged by an enclosing when!
        for a in lengthwrites(args)                 # arguments of the form lengthof(a); a is resizable
            addnode!(g, PrepareNode(a, LengthPrepare(), nothing))   # writes the launch commands and sizes over a
        end
    end
    return nothing
end
dispatch!(g::Graph, args...; kw...) = dispatch!(g.location, args...; kw...)

# A graph is never replaced for a structure change (Simon, 2026-10-05: "nen
# plot hinzufuegen darf niemals den kompletten graph wegschmeissen"; decisions
# B3.16). Nodes declared into a compiled graph, or deleted from it, only mark
# its plan: the next run!(g) compiles a new plan from the graph's nodes and
# retires the old one after its last run (recompile!). The graph, its windows,
# the resources it names and the device arrays it reads stay. Nodes that will
# be removed are declared in a block that returns a handle (names open):
#   h = insert!(g) do … end     # a plot's draws
#   delete!(g, h)               # the plot removed
# Nodes declared outside a block stay until free!(g). A graph array's
# contents live only during a run, so what must survive runs and recompiles
# (a film, an accumulation count) is a device array the graph names.
# Declaring takes the graph lock, so it waits while a run of g has stages
# left; a declaration inside another declaration of the same task (a
# repeat! body, an insert! block) runs under the lock already held. A host
# function of g that declares into g throws (lockgraph), as does an insert!
# inside a repeat! or when! body.
mutable struct NodeBlock
    const graph::Graph
    nodes::Vector{Node}             # its top-level nodes (a repeat! is one LoopNode)
    live::Bool                      # false once deleted: a second delete! throws
end
declarable(g::Graph) = (@atomic g.freed) && throw(ArgumentError("declaring into a freed graph"))
declaring(f, g::Graph) = g.declarer === current_task() ? f() : lockgraph(g) do
    declarable(g)
    g.declarer = current_task()
    try f() finally g.declarer = nothing end
end
# A top-level node names its resources (a LoopNode those of its body), joins
# the open block and marks a compiled graph. A repeat! body's nodes are added
# to its LoopNode instead.
function addnode!(g::Graph, n::Node)
    push!(g.nodes, n)
    g.inloop && return
    name!(g, n)
    g.block === nothing || push!(g.block.nodes, n)
    g.plan === nothing || (@atomic g.recompile = true)
end
# One node from a declaring call (build!, refit!, a render pass, a copy).
declare!(g::Graph, n::Node) = declaring(() -> (foreach(usable, persistentresources(n)); addnode!(g, n)), g)

function Base.insert!(f, g::Graph)
    g.inbody && throw(ArgumentError("insert! inside a repeat! or when! body"))
    declaring(g) do
        g.block === nothing || throw(ArgumentError("insert! blocks do not nest"))
        h = NodeBlock(g, Node[], true)
        g.block = h
        ok = false
        try
            f(); ok = true
        finally
            g.block = nothing
            ok || removenodes!(g, h.nodes)          # a throw inside the block leaves the graph as it was
        end
        return h
    end
end
Base.delete!(g::Graph, h::NodeBlock) = declaring(g) do
    h.graph === g && h.live || throw(ArgumentError("delete! of a node block not in this graph (deleted already?)"))
    h.live = false
    removenodes!(g, h.nodes)
end
# A resource no remaining node names leaves `named`; while a plan names it,
# its hold drops only after that plan's last run (recompile!; release!).
function removenodes!(g::Graph, nodes)
    drop = IdSet(nodes)
    filter!(n -> !(n in drop), g.nodes)
    unname!(g, (x for n in drop for x in declared(n)))
    empty!(nodes)
    g.plan === nothing || (@atomic g.recompile = true)
end
declared(n::Node) = (n,)
declared(p::RenderNode) = [p; [d for piece in p.pieces for d in piece.draws]]   # targets, then each draw
function unname!(g::Graph, nodes)
    for n in nodes, st in unique(persistentresources(n))   # a LoopNode: its body's and its count array
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

# copyto! between arrays in a graph is a copy kernel, so it may sit in a loop body.
Base.copyto!(dst::MantleArray, src::MantleArray) =
    (dispatch!(location(dst, src), copy_kernel!, (dst, src), launchrange(src)); dst)

# repeat!: the body is declared once and may only launch kernels and traces.
inloopbody(::KernelNode) = true
inloopbody(::TraceNode) = true      # indirect; the loop head zeroes its count once the loop ends
inloopbody(::PrepareNode) = true
inloopbody(::Node) = false

repeat!(body, g::Graph, count::Int; while_nonzero = nothing) = declaring(g) do
    foreach(usable, resources((while_nonzero,)))
    first = length(g.nodes) + 1
    g.inbody = true; g.inloop = true   # selection offers kernel candidates only (section 3)
    nodes = try
        body(LoopIndex())           # a placeholder argument; the loop head writes i into its slots
        g.nodes[first:end]
    finally
        g.inbody = false; g.inloop = false
        resize!(g.nodes, first - 1) # the body's nodes leave the graph's list in any case (named with the LoopNode)
    end
    all(inloopbody, nodes) ||
        throw(ArgumentError("a repeat! body launches kernels and traces only; found $(nodetypes(nodes))"))
    addnode!(g, LoopNode(count, while_nonzero, nodes, Origin(repeat!, (count,)), nothing))
end
# when!: every launch declared in the block runs only where `flag` is nonzero on
# the GPU (Simon, 2026-10-02). Launches only, like repeat!; blocks do not nest;
# a when! block may sit in a repeat! body, so inbody is restored, not cleared.
# dispatch! and trace! read g.condition and tag their node with it.
when!(body, g::Graph, flag) = declaring(g) do
    g.condition === nothing || throw(ArgumentError("when! blocks do not nest"))
    usable(flag)                    # named through its nodes, which carry it as their condition
    first = length(g.nodes) + 1
    outer = g.inbody
    g.inbody = true; g.condition = flag
    kept = false
    try
        body()
        nodes = g.nodes[first:end]
        all(inloopbody, nodes) ||
            throw(ArgumentError("a when! block launches kernels and traces only; found $(nodetypes(nodes))"))
        kept = true
    finally
        g.inbody = outer; g.condition = nothing
        kept || (g.inloop ? resize!(g.nodes, first - 1) :      # inside a repeat! body: nothing was named
                            removenodes!(g, g.nodes[first:end]))   # a throw leaves the graph unchanged
    end
    return nothing
end
# Inside render!, when!(p, flag) do … end tags the draws declared in the block
# the same way; a draw's visibility is a store into its flag, never a new plan.
# The mechanism repeat!'s loop head uses: every launch and draw in the block
# is indirect, and whoever writes its indirect command multiplies the count by
# the flag. That is the head for a flag the host writes (a store or a value
# GPURef), and a PrepareNode(flag, FlagPrepare()) right after the kernel that
# writes it otherwise (compile inserts it, section 4). No conditional
# rendering: the flag is read by address like any argument, so it may move
# and is never a handle dependence, and Metal, which has no conditional
# rendering, needs nothing else. CUDA and HIP have no indirect launch; their
# device types take the condition their own way (a conditional IF node; a
# check at kernel entry on HIP), from the same flag (plan, device-type table).

# A host function inside a call is a node; Mantle schedules it (section 6).
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
# Inside a repeat! body or a conditional only kernel candidates qualify: a
# foreign node (cuBLAS, MPSGraph) cannot run under a device-decided condition.
candidatesfor(::Device) = AnyCandidate()
candidatesfor(l::GraphDevice) = graph(l).inbody ? KernelsOnly() : AnyCandidate()
function select(dev::Device, allowed, f, args...)
    key = SelectionKey(f, args...)            # function, element types, shapes (capacity for a resizable array), layout, alignment
    measured = get(selectiontable(dev), (key, allowed), nothing)
    measured === nothing || return measured
    cands = [k for k in candidates(dev, f) if supports(k, key) && qualifies(k, allowed)]
    return argmin(k -> estimate(dev, k, key), cands)
end
qualifies(k, ::AnyCandidate) = true
qualifies(k::KernelCandidate, ::KernelsOnly) = true
qualifies(k, ::KernelsOnly) = false

# ══ 4. Compile ════════════════════════════════════════════════════════════════

# A fixed sequence of steps, each a function of the compile state. Record and
# run may not inspect the graph or the device; this is where that happens.
# The state holds copies of the graph's nodes (a loop's body copied too): the
# plan owns them and their compiled data.
function compile(g::Graph)
    s = CompileState(g, [deepcopynode(n) for n in g.nodes])
    pushfirst!(s.nodes, HeadNode(nothing, nothing))
    for h in hostnodes(s.nodes)     # each later stage copies the cells again
        insertafter!(s.nodes, h, HeadNode(h, nothing))
    end
    for (w, flag) in flagwriters(s.nodes)           # a when! flag a kernel of this graph writes:
        insertafter!(s.nodes, w, PrepareNode(flag, FlagPrepare(), nothing))   # its block's commands × flag (section 2)
    end
    layout!(s)                      # the entry and the Bound resources, before anything reads them
    isempty(g.rewrites) || (dependencies!(s); foreach(r -> r(s), g.rewrites))
    dependencies!(s)
    foreach(n -> compile!(s, n), s.nodes)
    foreach(f! -> f!(s), (costs!, schedule!, hostcuts!, presentspan!, liveness!, place!, sharing!,
                          barriers!, submissions!, summary!, bindings!))
    return Plan(s)      # stages, placement, argument block, entry, tables;
end                     #   starts marked: unmapped, restore, entrychanged; running = false

# What each node touches. The Bound resources are the plan's argument slots.
uses(p::KernelNode, s) = (argumentuses(p.args)..., (Bound(r) => Read() for r in bound(p.args))..., flaguses(p.when)...)
uses(p::TraceNode, s) = uses(p, s, p.args)
uses(p::LoopNode, s) = (flaguses(p.flag)..., (u for q in p.body for u in uses(q, s))...)
uses(p::HeadNode, s) = (                    # everything a head reads and writes
    headafter(p.after)...,                                 # a later stage's head runs after its host node
    (Bound(r) => Write() for r in boundresources(s))...,   # addresses, sizes, launch commands, GPURef values
    (r => CellRead() for r in celled(s))...,               # each device or resizable array's cell
    s.entry => Read(),
    (x => Write() for x in hostwritten(s, p))...)          # hostwritten data copied into the arena (the first head)
headafter(::Nothing) = ()
headafter(h::HostNode) = (HostDone(h) => Read(),)
uses(p::PrepareNode, s) = uses(p, s, p.kind)
uses(p::PrepareNode, s, ::LengthPrepare) = (p.array => CellRead(), Bound(p.array) => Write())
uses(p::PrepareNode, s, ::FlagPrepare) = (p.array => Read(), (Bound(q) => Write() for q in conditionedby(s, p.array))...)
uses(p::HostNode, s) = (hostreads(p.args)..., (w => Write() for w in p.writes)..., HostDone(p) => Write())
# Members compile does not know (push!, delete! and rebinds change them
# without recompiling) get placeholders. A node that reads through a
# composite (a build or refit of a structure, a trace or ray query of a TLAS,
# whose hit shaders read BLAS geometry, a launch or draw reading through an
# array GPURef) also reads AnyWrite: it conflicts with every earlier write in
# the plan, and every later write conflicts with it. A kernel that writes
# through an array GPURef writes AnyMemory, a write every later node
# conflicts with. dependencies! and barriers! match both against every key,
# not by name, which gives the edges and a full memory barrier. At run the
# members are known: expand (B.4) turns the build into a write of the
# structure and reads of its members, a read of a composite into reads of its
# members, and the WriteThrough of a GPURef into a read of it and a write of
# its target, for waits, records and the stage's written arrays.
uses(p::BuildNode, s) = (p.structure => BuildUse(), AnyWrite() => Read())
argumentuse(r::ArrayBox, ::Write) = (r => WriteThrough(), AnyMemory() => Write())   # inside argumentuses
argumentuse(r::ArrayBox, ::Read)  = (r => Read(), AnyWrite() => Read())
argumentuse(t::TLAS, ::Read)      = (t => Read(), AnyWrite() => Read())             # traces and ray queries

# ── dependencies!: an edge "q before p" wherever two nodes conflict ──
# One hazard function, shared with barriers! and the ordering across graphs.
hazard(before::Use, after::Use) =
    !(reads(before) && reads(after)) && !(unordered(before) && unordered(after))

function dependencies!(s)
    last = Dict{Any,Vector{Tuple{Node,Use}}}()
    for p in s.nodes, (r, u) in uses(p, s)
        for (q, v) in earlier(last, r)
            hazard(v, u) && addedge!(s.dag, q, p)
        end
        push!(get!(last, r, []), (p, u))
    end
    # The head writes every Bound and every launch reads its own, so the head
    # precedes every root without a special case. Keep only direct predecessors
    # (what a CUDA/HIP graph node takes), but never drop an edge whose only
    # replacement runs through a conditional node: that node may not run.
    transitivereduce!(s.dag; through = n -> !conditional(n))
end
# The earlier uses a use conflicts with: those of the same key, an earlier
# write through a GPURef (AnyMemory, written anything), and an earlier read
# through a composite (AnyWrite, read anything: a later write must follow
# it); for AnyWrite and AnyMemory themselves, every earlier use of any key, of
# which hazard keeps the conflicting ones.
earlier(last, r) = [get(last, r, ()); get(last, AnyMemory(), ()); get(last, AnyWrite(), ())]
earlier(last, ::Union{AnyWrite,AnyMemory}) = [e for es in values(last) for e in es]

# ── compile!: one method per node type; no lowering here ──
# CompiledDispatch(kernel, extents, arguments): arguments are InBlock() (the
# plan's argument block at the node's slots) in a graph, the packed region in
# an eager call (section 11). launchextents is Indirect for a node inside a
# when! block (its command's count is multiplied by the flag, section 2).
function compile!(s, p::KernelNode)
    p.compiled = CompiledNode()
    p.compiled.command = CompiledDispatch(kernel!(s.device, p.kernel, argtypes(p.args)),
                                          launchextents(s.device, p), InBlock())
end
function compile!(s, p::TraceNode)
    p.compiled = CompiledNode()
    p.compiled.command = CompiledTrace(compiletrace(s.device, p.pipeline, argtypes(p.args)),
                                       launchextents(s.device, p), InBlock())
end
# A render pass: its targets and their load operations, one compiled draw per
# draw. Pipelines are values; the device type compiles each for the argument
# types and the target formats (cached by the device type, like kernels).
function compile!(s, p::RenderNode)
    p.compiled = CompiledNode()
    draws = [CompiledDraw(compiledraw(s.device, d.pipeline, argtypes(d.args), formats(p.targets)),
                          drawextents(s.device, d), InBlock()) for d in p.draws]
    p.compiled.command = CompiledRender(p.targets, draws)
end
function compile!(s, p::LoopNode)
    p.compiled = CompiledNode()
    foreach(q -> compile!(s, q), p.body)
    # The loop start restores the body's launch commands (a previous run's loop
    # head may have zeroed them); the loop head evaluates the flag, zeroes the
    # body's indirect counts once it is 0, and writes i into the body slots
    # that take it.
    p.compiled.command = CompiledLoop(kernel!(s.device, loopstart_kernel!, loopstarttypes(p)),
                                      kernel!(s.device, loophead_kernel!, loopheadtypes(p)), p.count, p.flag)
    derivebodyhazards!(s, p)        # last body node → first body node of the next iteration
end
compile!(s, p::HeadNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledHead(headkernels(s.device)...))
compile!(s, p::PrepareNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledDispatch(kernel!(s.device, prepare_kernel!, preparetypes(p)), 1, InBlock()))
compile!(s, p::HostNode) = (p.compiled = CompiledNode(); p.compiled.command = CompiledHost(p.f, p.args))

# Core keeps no kernel cache: the device type answers from its compiler's cache
# and says whether this call compiled, for the per-device compile counter.
function kernel!(dev, f, types)
    native, compiled = compilekernel(dev, f, types, config(dev, f))
    compiled && increment!(dev.compiles)
    return Kernel(native, kernelinfo(dev, native))
end

# ── costs!: estimated, never measured-then-re-recorded ──
costs!(s) = foreach(p -> (p.compiled.cost = estimate(s.device, p)), s.nodes)
estimate(dev, p) = estimate(caps(dev), p) * calibration(dev, p)
estimate(c::Caps, p) = max(work(p) / c.throughput[eltype(p)], bytes(p) / c.bandwidth)
    # work: 2MNK for a mul!, else from KernelInfo and the ndrange (the capacity for a kernel-written length)
    # c.throughput, c.bandwidth: facts the device type reports (api.md G3)
    # calibration: from past profiled runs of this device, 1.0 without any

# ── schedule!: order and queue per node ──
function schedule!(s)
    s.order = toposort(s.dag)
    foreach(p -> (p.compiled.queue = queuefor(s.device, p)), s.order)
end
queuefor(dev, ::HostNode) = HostSide()
queuefor(dev, ::HeadNode) = channel(dev, Main())
queuefor(dev, ::VideoDecodeNode) = channel(dev, VideoDecode())
queuefor(dev, p::CopyNode) = queuefor(dev, p, independenthostcopy(p), caps(dev).transfer)
queuefor(dev, p::CopyNode, ::Independent, ::HasTransfer) = channel(dev, Transfer())
queuefor(dev, p::Node) = overlapgain(p) > caps(dev).crossqueuecost ? channel(dev, Compute()) : channel(dev, Main())

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
        compile!(s, c); c.compiled.cost = estimate(s.device, c); c.compiled.queue = queuefor(s.device, c)
    end
end

# ── liveness!, place!: placement stored in the plan ──
function liveness!(s)
    for (i, p) in enumerate(s.order), r in transients(p)
        s.first[r] = min(get(s.first, r, i), i)
        s.last[r]  = max(get(s.last, r, i), i)
    end
    for x in hostwritten(s)
        s.first[x] = 1              # the head writes it at the start of the run
    end
    # a loop body's transients live for one iteration unless read before written;
    # a hostwritten one read in a body lives for the whole loop
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
# memkey(s, AnyWrite()) and memkey(s, AnyMemory()) overlap every key: the node
# gets a global memory barrier (VkMemoryBarrier2 without a resource, one per
# build node; device memory caches only, no layout change).

# ── submissions!: stages of segments, cut by budget, queue, host nodes, present ──
# A stage is the segments between two host nodes, and the host node after them.
function submissions!(s)
    c = Cutter(submissionbudget(caps(s.device)))    # core's policy from the device's facts: a smaller target
                                                     #   where caps(dev).display (new work: today 0.1 s for all)
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
    (c.segment.channel !== p.compiled.queue || c.time + p.compiled.cost > c.budget || writeswindow(p))
added!(c, p::Node) = writeswindow(p) && (c.segment.presents = true)
added!(c, p::PresentNode) = closesegment!(c)      # the present segment ends here; the next node opens a new one
# The present segment is one submission on one queue: from the first node that
# writes the swapchain image to the PresentNode. presentspan! runs after
# hostcuts! (whose copies may land in the span): every node in the span goes
# on the present channel; a BuildNode or HostNode in it that nothing in the
# span depends on, and that depends on nothing in it, moves before the span;
# one that cannot move is a compile error (a per-image recording holds no
# AccelCommand, and the image cannot wait across a host function).
# A node moves out when it does not depend, directly or transitively, on the
# first window write (judged per node in order, so a node depending only on
# nodes that moved out moves too); everything else stays and goes on the
# present channel.
function presentspan!(s)
    span = presentspanof(s)                          # empty without a window
    w = first(span)
    for p in span[2:end]
        dependson(s.dag, p, w) ? setqueue!(p, presentchannel(s)) : movebefore!(s, p, w)
    end
    any(p -> notinpresent(p), presentspanof(s)) &&
        throw(ArgumentError("a build or host node between rendering into the window and presenting it"))
end
notinpresent(::Union{BuildNode,HostNode}) = true
notinpresent(_) = false

# A loop is cut between iterations when one iteration fits the budget and the
# loop does not (open decision 11).

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
    # cell), persistent graph arrays, images, acceleration structures, array
    # GPURefs, arena parts, the plan's entry. Bound resources are the plan's
    # own. A composite (a TLAS, a BLAS, an array GPURef) is listed as itself;
    # its members are added at run, from the snapshots lockexpanded takes
    # (resizing-and-raytracing.jl B.4), since a rebind or push! changes them
    # without recompiling.
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

struct MergeRegions; attribute::Any; end
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
        b = Builder(pl.device, seg.channel)          # command objects from the device's free list
        for p in seg.nodes
            b = emitpart!(b, seg, pl, p)             # the builder to continue in
        end
        push!(seg.parts, Recording(finish(b), seg.channel))
    end
end
# Record compiles nothing, allocates nothing, submits nothing, reads no mutable
# resource field: everything it writes comes from the compiled nodes. A
# segment is a list of parts submitted together: recordings, written here
# once, and AccelCommands. A run re-records only the AccelCommands a change
# marked (resizing-and-raytracing.jl B.8) and, per swapchain generation, the
# segments that draw into window-sized targets and the per-image present
# segments (section 7).

emitpart!(b, seg, pl, p::Node) = (emit!(b, pl, p); b)
# A build or refit ends the part on every device type: it goes into an
# AccelCommand, its own small command buffer that the first run records with
# the same builder verbs, and that a run records again only when a change
# marked it (a new count, a MoveAccel; B.8). Its barrier is recorded with it.
function emitpart!(b, seg, pl, p::BuildNode)
    push!(seg.parts, Recording(finish(b), seg.channel), AccelCommand(p.structure, p.mode, deps(p), seg.channel))
    return Builder(pl.device, seg.channel)
end

emit!(b, pl, p::KernelNode)  = emitdispatch!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::TraceNode)   = emittrace!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::PrepareNode) = emitdispatch!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::CopyNode)    = emitcopy!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::ForeignNode) = emitlibrary!(b, p, p.compiled.command, deps(p))
emit!(b, pl, p::RegionNode)  = emitlibrary!(b, p, p.compiled.command, deps(p))   # a region compiler's output
emit!(b, pl, p::LoopNode)    =
    emitloop!(b, p, p.compiled.command, deps(p)) do
        foreach(q -> emit!(b, pl, q), p.body)
    end
# Draws inside a render pass carry no barrier: the pass's barrier (target
# layouts, and everything its draws read) is beginrender!'s. The pass is its
# list of pieces, in draw order (below).
function emit!(b, pl, p::RenderNode)
    beginrender!(b, p, p.compiled.command, deps(p))   # the fixed barrier included
    emitpieces!(b, p, [cp.native for cp in piecesof(pl, p)])   # the plan's list (updatepieces!); device-type verb
    endrender!(b)
end

# ── Pieces: draws inserted into a compiled pass (decisions B3.17) ──
# Simon, 2026-10-05: optimize complex GUIs without slowing large meshes and
# scatters; an insert in the millisecond range is fine, and visibility must
# be cheap. A render pass is an ordered list of pieces. A piece is the draws
# of one block: the draws declared in render!'s do-block are the first, and
# each `h = insert!(p) do … end` on the pass adds one (a plot). A piece is
# compiled and recorded on its own, so inserting or deleting one compiles and
# records that piece and rewrites the pass's list (one part recorded again
# before its next submission, as a marked AccelCommand); the plan is not
# recompiled and no other piece is recorded again. Each draw keeps its own
# pipeline, index buffer and argument layout, so a large mesh draws exactly
# as it would alone. Visibility stays a store into the draw's when! flag (the
# count times the flag): no piece is touched. A changed draw order is a
# rewrite of the list (open decision 15, option (c) at the cost of one list).
#
# What makes a piece independent of the plan:
#   - its own argument region (a pool region the piece holds) and its own
#     entries in the head's cell-copy table, which is a resizable array of
#     the plan: the head launches over its length (indirect), so new entries
#     record nothing; a deleted piece's entries are blanked and reused, and
#     the table is compacted at the next recompile;
#   - a fixed barrier at the start of every pass (compute, copy and build
#     writes before it -> vertex, index, indirect and fragment reads), so a
#     new draw adds no dependency inside the graph that compile would have
#     to place;
#   - its own registration: the piece's resources join the namers (B3.15),
#     holds and dependents, as register! does for a plan, for these uses only.
# A piece holds draws only (when! blocks of draws included); a block that
# declares anything else throws, and kernels per plot go into the graph with
# insert!(g) (B3.16), which recompiles. Core Vulkan inherits no state into a
# secondary command buffer ("No state, including dynamic state, is inherited
# from one command buffer to another", pipelines.adoc:131-133), so each piece
# binds its pipelines and sets viewport and scissor itself: the pieces of a
# pass into a window are recorded per swapchain generation, like the other
# window-sized parts (a resize records each piece once).
#
# Two lists. The graph's pass node holds the declared pieces (insert! and
# delete! edit it under the graph lock; a recompile compiles from it). Each
# plan holds its own compiled list per pass (piecesof), which only
# updatepieces! edits, under the graph lock before the run (run!(g)): an
# inserted piece joins it once compiled and recorded, a deleted one leaves it
# before its region and recording are retired. recordwindows! records the
# plan's list, after updatepieces! (run!(g)).
mutable struct Piece                # declared: the graph's
    const pass::RenderNode
    draws::Vector{DrawNode}         # in declaration order; each names its resources (holds counted per draw)
    live::Bool                      # false once deleted (a second delete!, or an insert! after it, throws)
end
mutable struct CompiledPiece        # one plan's; released with the plan, or by a delete
    const piece::Piece
    args::Region                    # its argument region (value GPURef slots included, marked per piece)
    copies::Vector{Int}             # its entries in the plan's cell-copy table
    native::Any                     # the device type's recording (recordpiece), per generation for a window pass
end

# render! returns the pass; its do-block declares the first piece. draw!
# appends to the open piece and names the draw's resources (holds counted per
# draw, like nodes).
function render!(f, g::Graph, targets...)
    declaring(g) do
        p = RenderNode(targets, Piece[], nothing, Origin(render!, targets), nothing)
        first = Piece(p, DrawNode[], true)
        p.open = first
        ok = false
        try
            f(p); ok = true
        finally
            p.open = nothing
            ok || unname!(g, first.draws)
        end
        push!(p.pieces, first)
        addnode!(g, p)                              # names the targets; a compiled graph recompiles (a new pass)
        return p
    end
end
function draw!(p::RenderNode, pipeline, args, count; instances = 1, indices = nothing)
    p.open === nothing && throw(ArgumentError("draw! outside render!'s block or an insert! block of the pass"))
    foreach(usable, resources((args, count, instances, indices)))
    d = DrawNode(pipeline, args, count, instances, indices, p.condition, Origin(draw!, args), nothing)
    push!(p.open.draws, d)
    graph(p).inloop || name!(graph(p), d)           # in a repeat! body a render! throws; nothing to unname then
end

# Declared under the graph lock like any declaration. Only the graph's next
# run compiles it (run!(g): updatepieces!), never the call; until then the
# pass draws without it.
function Base.insert!(f, p::RenderNode)
    g = graph(p)
    declaring(g) do
        p.open === nothing || throw(ArgumentError("insert! blocks do not nest"))
        p in g.nodes || throw(ArgumentError("insert! into a pass that was deleted from its graph"))
        piece = Piece(p, DrawNode[], true)
        p.open = piece                              # draw!(p, …) appends to the open piece and names its resources
        ok = false
        try
            f(); ok = true
        finally
            p.open = nothing
            ok || (unname!(g, piece.draws); empty!(piece.draws))
        end
        push!(p.pieces, piece)                      # the list order is the draw order
        g.plan === nothing || push!(g.plan.piecechanges, Inserted(piece))
        return piece
    end
end
Base.delete!(p::RenderNode, piece::Piece) = declaring(graph(p)) do
    piece.pass === p && piece.live || throw(ArgumentError("delete! of a piece not in this pass (deleted already?)"))
    piece.live = false
    filter!(x -> x !== piece, p.pieces)
    g = graph(p)
    g.plan === nothing ? unname!(g, piece.draws) : push!(g.plan.piecechanges, Deleted(piece))
end

# Applied by run!(g) under the graph lock, before the run's stages, in call
# order. An insert: pipelines from the device type's cache, the argument
# region (acquire!, before any lock, like prepare!), the cell-copy entries
# (a store into the plan's table, applied by the run's first stage), the
# piece's recording (recordpiece) and its registration. A delete: the piece
# leaves the namers (its last-run tokens into the resources' `retired`, the
# other namers notified) and its holds drop, its region, entries and
# recording are retired with the plan's last tokens (the graph lock is held,
# so no run of this plan is submitting). Either way the part holding the pass
# is marked and recorded again before its next submission. A piece that
# fails to compile (a pipeline error, memory) is taken out of the pass, its
# resources unnamed, and run! throws; the rest of the graph keeps running.
function updatepieces!(pl::Plan)
    passes = unique(c.piece.pass for c in pl.piecechanges)
    try
        foreach(c -> apply!(pl, c), pl.piecechanges)
    finally
        empty!(pl.piecechanges)                     # a failed insert was taken out; run! throws after this
        for pass in passes                          # the plan's list: the graph's order, compiled pieces only
            setpieces!(pl, pass, [compiledof(pl, x) for x in pass.pieces if iscompiled(pl, x)])
            markpart!(pl, pass)                     # the list part is recorded again before its next submission
        end
    end
end
function apply!(pl::Plan, c::Inserted)
    c.piece.live || return                          # deleted before it was ever compiled
    ok = false
    try
        cp = compilepiece!(pl, c.piece)             # pipelines, region, copy-table entries, recordpiece
        registerpiece!(pl, c.piece)                 # register! (resizing-and-raytracing.jl B.6) for its uses only;
        ok = true                                   #   a resource the plan names already gets the union of both uses
    finally
        ok || (c.piece.live = false; filter!(x -> x !== c.piece, c.piece.pass.pieces);
               releasepiece!(pl, c.piece, ()); dropunnamed!(graph(pl), c.piece.draws, ()))   # never used by a run
    end
end
function apply!(pl::Plan, c::Deleted)
    tokens = lasttokens(pl)
    dropunnamed!(graph(pl), c.piece.draws, tokens; plan = pl)
    releasepiece!(pl, c.piece, tokens)              # its CompiledPiece: region, copy-table entries (blanked), recording
end
# The resources no other node or piece names any more (unname! puts them in
# g.unnamed; a recompile ran first if one was due, so nothing else is there):
# the plan leaves their namers (its last-run tokens into their `retired`, the
# other namers notified, as unregister!), and their holds drop after `tokens`.
function dropunnamed!(g::Graph, draws, tokens; plan = nothing)
    unname!(g, draws)
    plan === nothing || unregisteruses!(plan, g.unnamed)
    foreach(st -> dropafter!(st, tokens), g.unnamed); empty!(g.unnamed)
end
# A recompile compiles every piece in the passes' lists with the new plan. It
# first unnames the draws of pieces deleted since the last run (their holds
# drop with the old plan's tokens, below) and clears piecechanges.
recompiled!(g, c::Deleted) = unname!(g, c.piece.draws)
recompiled!(g, ::Inserted) = nothing
# A PresentNode records only its barrier (the image to the present layout);
# the present itself is submitnative!'s (section 8). A VideoDecodeNode
# records through the decode verbs of the plan's video section.
emit!(b, pl, p::PresentNode) = emitcopy!(b, p, NoCopy(), deps(p))   # the barrier, no copy
# The head: applies the entry, copies cells into the Bound slots, computes the
# launch commands and ndrange values over resizable arrays and multiplies the
# counts inside a when! block by its flag where the host writes the flag
# (section 2). Its barrier to the
# launches after it is theirs (barriers! put it on the first reader of each Bound).
emit!(b, pl, p::HeadNode) =
    for k in p.compiled.command.kernels              # applyentry, copycells, launches
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
# graph; then the run starts over. Pieces are updated before the windows are
# recorded, so a new generation records the current list.
function run!(g::Graph)
    while true
        r = lockgraph(g) do
            (@atomic g.freed) && throw(ArgumentError("run! of a freed graph"))
            while true
                (g.plan === nothing || @atomic g.recompile) && recompile!(g)
                isempty(g.plan.piecechanges) || updatepieces!(g.plan)   # inserted or deleted pieces (section 5)
                applywindows!(g)    # a resize! of a window this graph renders to (section 7)
                v = run!(g.plan)
                again(v) || return v                # Submitted, Skipped or WaitFor
            end
        end
        done(r) && return r                         # Submitted() or Skipped(), both public
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
        foreach(c -> recompiled!(g, c), g.plan.piecechanges); empty!(g.plan.piecechanges)
    end
    ok = false
    new = nothing
    try
        new = compile(g)            # fallible work, all before any token is claimed
        record!(new)
        while !done(register!(new)) # a dependency changed during compile (B.6)
            discard!(new); new = nothing   # retired once: the finally below must not see it
            new = compile(g); record!(new)
        end
        for (w, sc, size) in currentgenerations(g)   # each captured under its window's lock (section 7)
            recordwindows!(new, w, sc, size)
        end
        writeentry!(new, g.pending, g.plan)   # the old entry's hostwritten ranges, then g.pending over them;
        empty!(g.pending)                     #   a new plan starts with entrychanged = true
        old, g.plan = g.plan, new
        ok = true                             # new is the graph's plan now: a throw below must not discard it
        old === nothing || retireplan!(old, new)   # B.6: off the
                                              #   dependents, value GPURefs, structures' command lists, held targets
        foreach(st -> dropafter!(st, lasttokens(old)), g.unnamed); empty!(g.unnamed)   # resources deleted nodes named
    finally
        # A failed compile leaves the mark; a plan it made is discarded, so no
        # dependents entry, command list, held target or arena tenancy
        # outlives it.
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
    pages::IdDict{Arena,Any}            # per arena part of the plan: pages for its unmapped range (section 9)
    prepared::Prepared                  # what applying the pending changes and restores needs (B.4 step 1)
    image::Any                          # nothing, an Image not yet presented, WindowChanged() or Minimized()
    generation::Any                     # the swapchain generation this run's window-sized segments use
    waits::Vector{Token}                # other graphs' work found by earlier stages (section 8); later stages wait too
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
# capacity, command objects for the AccelCommands it will record again, and
# restore storage for evicted resources when the plan is marked for restore;
# a store's upload region was acquired at its call), then under them
# submitwith! applies them. If what
# was prepared no longer fits (a change arrived meanwhile, during a host
# function say), the stage prepares again and retries; nothing of the stage
# was submitted, and earlier stages stay as they were.
function run!(pl::Plan)
    writeentry!(pl)                 # only after a value GPURef changed: waits for the previous run (below)
    sweep!(pool(pl.device))         # release what has passed (section 9)
    @atomic pl.running = true       # unmapidle! and eviction leave what this plan names alone until the run ends
    rs = RunState(IdDict(), Prepared(), nothing, nothing, Token[])
    held = false
    try
        # The image before the arena parts: its acquire may block (vsync, a
        # hidden window), and no other graph's run waits on this run's parts
        # meanwhile.
        rs.image = acquireimage!(pl)     # before any lock (section 7)
        skip(rs.image) && return Skipped()
        again(rs.image) && return Again()
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
                done(r) && break
                i == 1 && return r                   # Again or WaitFor: run!(g) starts over (waiting outside its lock)
                # A later stage never gets WaitFor (takepending leaves such
                # changes for after this run); Again: prepare and lock again.
            end
            pendreaders!(pl, r.written)               # after the stage is submitted (open decision 16, ruled B3.9/B3.10)
        end
        return Submitted()
    finally
        endbetween!(pl)                               # a throw between stages; nothing if already cleared
        foreach(((part, pages),) -> release!(pool(pl.device), pages), rs.pages)   # pages no change took
        release!(rs.prepared)                         # what no step took
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
    res = expanded([snap; extra])
    plans = lockset(pl, stage, res)
    r = lockplans(plans) do
        recs = [[syncstate(r) for r in res]; (i == 1 ? [a.sync for a in arenaparts(pl)] : []);
                [c.structure.sync for c in accelcommands(stage)]]   # what current! reads (B.8)
        locked(recs) do
            inboxsnapshot(pl) === snap || return Retry()
            lockset(pl, stage, res) ⊆ plans || return Retry()
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
        ownwaits = [waits[seg]; [s.token for s in seg.after]]
        # The parts' command buffers are taken only after the change prefix is
        # emitted (inside submitwith! for the first segment), so a marked
        # AccelCommand is recorded again with the builtcount the prefix's
        # Build just set (B.8).
        if k == 1
            r = submitwith!(seg.channel, recs, rs.prepared, (), parts; self = pl, ownwaits, present,
                            restores = stage === first(pl.stages) && (@atomic pl.restore))
            done(r) || return r                       # Again or WaitFor
            seg.token, changed = r
        else
            touched(seg, changed) && push!(ownwaits, first(stage.segments).token)   # reads what the changes wrote
            seg.token = submit!(seg.channel, map(p -> commandbuffer(p, ()), parts); waits = ownwaits, present)
        end
        foreach(p -> submitted!(p, seg.token), parts)
        pl.lastrun[seg.channel] = seg.token          # the latest per channel covers the earlier ones
        present === nothing || (rs.image = nothing)  # presented: no longer the window's to keep; the generation
                                                     #   is released at the run's end (run!'s finally)
    end
    # Still under the locks: what the stage wrote goes into the inboxes of
    # the other plans reading it (section 8).
    foreach(r -> notifyreaders!(r, pl), namedwrites(stage))
    # The shared arrays the stage writes, from compile; a write through an
    # array GPURef counts as a write of its current target.
    return StageDone(writtenresources(stage))
end

# Arena parts are shared between graphs: the memory is the largest tenant's,
# not the sum. A run holds its parts from before its first submission until
# its last stage is submitted, so another graph's run on the same part waits
# on the host for the whole run, host functions included; the GPU order after
# that comes from the tokens. Waiting is preferred over memory of its own
# (Simon, 2026-10-02). Parts are taken in id order, so two runs never wait on
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

# Proposal (open decision 16). A stage that wrote arrays something else is
# built from pends the readers' updates once it is submitted: a TLAS Update,
# a BLAS Update (a Build when the array is its index array), applied by the
# next submission that touches the structure, which waits for this stage's
# write through the sync states (B.4); a screen's redraw mark. `written`
# is the arrays the stage wrote, from its expanded uses (a target written
# through a GPURef included). Readers come from each array's snapshot
# (immutable, replaced under its lock; resizing-and-raytracing.jl B.2), so a
# structure built after compile counts too, and iterating needs no lock. A
# structure this plan builds or refits itself has its own command and is
# skipped. One method of pendupdate! per reader type; each goes through
# change!.
pendreaders!(pl, written) =
    for a in written, r in readers(a)
        r in builtby(pl) || pendupdate!(r, a)
    end
readers(st::ArrayState) = st.readers   # the snapshot
readers(x) = ()                        # a structure, an image, an arena part: nothing is built from them here

commandbuffer(cmd, steps) = cmd                     # a finished command buffer (an eager call, submitnow!)
commandbuffer(p::Recording, steps) = p.native
commandbuffer(c::AccelCommand, steps) = current!(c, steps)   # from the state after `steps`, under the structure's lock (B.8)
submitted!(::Recording, token) = nothing
submitted!(c::AccelCommand, token) = (c.lasttoken = token; built!(c.structure, c.recordedmode, c))   # a Build sets builtcount (B.8)

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

# Only a plan whose entry holds a value GPURef that changed since its last
# run writes the entry, and only that write waits for the previous run (runs
# never overlap, and the entry has one copy). A store into a value GPURef
# marks every plan holding it (section 7); the swap before the write means a
# store during the write marks the plan again, so the next run writes it.
# Hostwritten data was written by `x[:] = data` (section 7).
function writeentry!(pl::Plan)
    (@atomicswap pl.entrychanged = false) || return
    waitfor(lastrun(pl))
    for r in valuerefs(pl)
        lock(r.lock) do             # a multi-word value is never torn
            put!(pl.entry, slot(pl, r), r.value)
        end
    end
end
lastrun(pl) = values(pl.lastrun)   # Tokens, the latest per channel

# ══ 7. Changes ═══════════════════════════════════════════════════════════════
# Stores, resize!, TLAS changes and how a submission applies them:
# resizing-and-raytracing.jl B.3-B.8. Here: GPURef, hostwritten transients,
# windows.

# GPURef(dev, v) for a value: a host value box with its own lock; a run of a
# plan it marked copies it into the entry (section 6). For scalars a kernel
# takes by value; a value that changes every frame (a camera matrix) is a
# one-element array written by a store instead, which never waits on the host.
# (GPURef(dev, a::MantleArray) is the array form, ArrayBox,
# resizing-and-raytracing.jl B.8b: a pending cell write, not an entry value.)
mutable struct ValueBox{T} <: GPURef
    const device::Device
    value::T
    const plans::Vector{WeakPlan}   # plans whose entry holds it: added by register!, removed when a plan retires
    const lock::ReentrantLock       # a leaf
end
function Base.setindex!(r::ValueBox, v)
    lock(r.lock) do
        r.value = v
        foreach(markentry!, r.plans)    # @atomic plan.entrychanged = true, through the weak reference
    end
end

# Hostwritten transient: into the entry after the previous run, or, while the
# graph has no current plan (before its first compile, or with a recompile
# pending), kept on the host until the next compile writes it into the new entry.
function Base.setindex!(x::MantleArray{T,N,GraphDevice}, data, ::Colon) where {T,N}
    ishostwritten(x) || nodata()
    g = graph(location(x))
    lockgraph(g) do
        if g.plan === nothing || @atomic g.recompile
            g.pending[x] = copy(data)
        else
            waitfor(lastrun(g.plan))                          # this call waits, not run!
            put!(g.plan.entry, entryrange(g.plan, x), data)   # the head copies it into the arena
        end
    end
end

# Window-sized arrays take any dims within the size their slots were placed
# for (the window's capacity: the largest monitor's size unless given); beyond
# it the graph recompiles at its next run. Their contents after a resize are
# undefined. Window-sized images are created again per swapchain generation,
# at the window's size, on the same arena memory: the arena part holding
# window-sized transients is sparse where the device has sparse residency,
# and a generation maps only the pages its size needs, so the capacity costs
# address space, not memory. Without sparse residency the capacity is mapped.

# One graph presents to a window (Simon, 2026-10-05, decisions.md B3.8): the
# graph that declares render! into it; a second graph declaring into it throws
# at declaration (other graphs render into Images this graph reads). So every
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
# recordings); that run's tokens release it, and its per-image recordings
# with it. A new generation retires the current swapchain (created with it as
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
        sc, size = lock(w.lock) do
            w.pendingsize === nothing || any(iszero, w.pendingsize) || newgeneration!(w)
            (w.swapchain, w.size)
        end
        foreach(x -> resizeheld!(x, size), windowsized(g, w))  # pending cell writes, or a recompile mark (graph lock held)
        # This generation's recordings, if the plan has none for it yet: the
        # window-sized images at this size, the segments that draw into them
        # (render areas and image handles are recorded by value), and the
        # per-image present segments. The previous generation's are retired
        # with its holds. recordwindows! also registers the AccelCommands of
        # those segments in their structures' command lists (each under its
        # structure's lock, as register! does), so a change marks them, and
        # removes the retired generation's; and it sets the plan's tenancy in
        # the window part for this size, marking the part when it grew
        # (section 9).
        g.plan === nothing || recordedfor(g.plan, sc) || recordwindows!(g.plan, w, sc, size)
        ok = true
    finally
        ok || (@atomic g.windowchanged = true)
    end
end
function newgeneration!(w::Window)   # under w.lock; no acquire in progress (the graph lock is held)
    old = w.swapchain
    w.size, w.pendingsize = w.pendingsize, nothing
    w.swapchain = createswapchain(w, w.size...; oldswapchain = old.native)
    foreach(i -> drophold!(i.swapchain), w.kept); empty!(w.kept)   # kept images of the old generation go with it
    drophold!(old)                   # retired with its runs' tokens once they have submitted
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
releasegeneration!(sc::Swapchain, tokens) = retirehold!(sc, tokens)
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
struct Token
    channel::SubmitChannel
    value::UInt64
end
passed(t::Token) = passed(t.channel, t.value)    # device-type verb: passed(ch, value)
waitfor(t::Token) = waitfor(t.channel, t.value)  # device-type verb: waitfor(ch, value)
waitfor(ts) = foreach(waitfor, ts)               # a collection of tokens
# A host wait is a driver call made with `@ccall gc_safe = true` (Julia 1.12,
# base/c.jl:278), so a thread waiting on the GPU does not hold up a GC another
# thread needs.

# ── Between graphs: through the resource, when it is written ──
# Open decision 19, option B (Simon, 2026-10-05: few recompiles, a cheap run).
# No plan keeps a list of other plans. A resource's SyncState has `namers`:
# the plans naming it with their use from compile (read, write or both), an
# immutable snapshot replaced under its lock by register! and unregister!
# (resizing-and-raytracing.jl B.6). A plan naming a composite reads the
# composite's members: namingplanuses(r) is r's namers and, as readers, the
# plans naming the composites built from r (its readers snapshot, B.2),
# transitively through array GPURefs. Snapshot reads, no lock.
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
# Both wait for the other plan's whole last run (Simon: waiting for the
# complete run is fine here). The waits are kept in the run's state and every
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
# to it). A plan that stops reading a resource through a composite (delete!,
# a rebind) stays among its namers until the submission that removed it has
# passed (resizing-and-raytracing.jl B.8); until then writers wait for it.
# A retired plan's last run is in the `retired` list of every resource it
# named, members of the composites it named included, and those resources
# are put into the other namers' inboxes (unregister!).
namingplanuses(r) = [namers(syncstate(r)); [(pl, through(c, u)) for c in readers(r) for (pl, u) in namingplanuses(c)]]
# A plan writing through an array GPURef writes its target; a structure or a
# screen built from r only reads it.
through(::ArrayBox, u) = u
through(_, u) = Read()
namingplans(r) = unique(first.(namingplanuses(r)))
readingplans(r) = unique(pl for (pl, u) in namingplanuses(r) if reads(u))
sharers(pl, stage) = unique(q for r in namedwrites(stage) for q in namingplans(r) if q !== pl)
# namedwrites(stage): from compile. A write through an array GPURef counts
# for the target its cell points at (`applied`) and, while a Rebind is
# pending, for r[] too: a stage that applies the Rebind in its prefix writes
# the new target, so it waits for and locks that target's other users as
# well. Both only change under the locks the stage holds.

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
# stages write and with its inbox, not with the number of resources it names
# (Simon, 2026-10-05).
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
    lock(ch.lock) do
        token = Token(ch, nexttoken!(ch))
        recordeager!(eager, token; self)
        submitnative!(ch, token.value, parts, waits, signals, present)   # the device type's verb
        return token
    end
end

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
function lockresources(f, uses)
    while true
        plans = namingplanssnapshot(uses)
        r = lockplans(plans) do
            lockexpanded(uses) do recs, x
                namingplanssnapshot(uses) === plans || return Retry()
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
# the addresses its arguments will have. Returns the token.
function submitapplying(own, uses; ch = channel(device(first(first(uses))), Main()))   # uses: (resource, use) pairs
    while true
        prepared = Prepared()
        r = nothing
        try
            prepare!(prepared, uses)                  # storage, pages, capacity, command objects, restores
            r = lockresources(uses) do recs, x       # the naming plans' sync locks, then the SyncStates
                submitwith!(ch, recs, prepared, expand(x, uses), OwnWork(own))   # records eager accesses
            end
        finally
            release!(prepared)                        # what no committed step took
        end
        done(r) && return first(r)
        waitforruns(r)                                # WaitFor: outside every lock; Again: nothing
    end
end
recordown(parts::Vector, steps) = map(p -> commandbuffer(p, steps), parts)   # a run's recorded parts
recordown(w::OwnWork, steps) = w.own(storageafter(steps))            # an eager call, a copy of its own
# A submission whose own work is a copy (a host read, an eviction copy): the
# same path, the copy emitted by `emit(b, after)`.
submitnow!(emit, uses; ch = channel(device(first(first(uses))), Main())) =
    submitapplying(after -> (b = Builder(device(ch), ch); emit(b, after); [finish(b)]), uses; ch)

# ══ 9. Memory: pool, kinds, arenas, release ═══════════════════════════════════

# Never called with an sync-state, channel or pool lock held.
function acquire!(pool::Pool, kind, bytes; constraint = nothing)
    sweep!(pool)
    r = tryacquire!(pool, kind, bytes, constraint)
    r === nothing || return r
    # Pressure is handled here and only here. It submits only the unmapping of
    # idle arena pages and eviction copies, and never flushes a queue.
    for relieve in (unmapidle!, trimblocks!, collectgarbage!)
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
    while true                                        # open decision 16, ruled B3.9/B3.10
        tokens = evictfor!(pool, bytes)
        isempty(tokens) && break
        waitfor(tokens)                               # outside every lock: the evicted storage's retirement tokens
        sweep!(pool)
        r = tryacquire!(pool, kind, bytes, constraint)
        r === nothing || return r
    end
    throw(OutOfDeviceMemory(kind, bytes, budget(pool.device)))
end

# ── Eviction and restore (open decision 16, ruled B3.9/B3.10) ──
# Arrays and images only. Acceleration structures are not evicted (a TLAS or
# BLAS would need its sources back before a rebuild).

struct Evicted
    host::Region              # Readback-kind memory with the contents, valid once `token` passed
    token::Token              # the eviction copy's submission
end

# Checked under the record's lock. Not evictable: anything a composite holds
# (a TLAS or BLAS range, an array GPURef reads it through a cell that a
# restore would have to reach), anything a plan recorded by handle (a restore
# would recompile it), anything a running plan names, lent or with pending
# changes. Accesses still in flight are allowed: the eviction copy waits for
# the last write through the sync state (a read need not wait for reads),
# and the storage is retired with the tokens of every access in flight and
# the copy's. So `run!(dnn_1); run!(dnn_2)` back to back evicts dnn_1's
# weights while its run is still on the GPU, and acquire! waits for those
# tokens. A structure's own arrays (records, range table, offsets, software
# storage) are held by it with composite holds, so they are never evicted.
evictable(st) = st.composites[] == 0 && !any(d -> d.handle, values(st.dependents)) &&
                !any(running, keys(st.dependents)) && !st.sync.lent &&
                isempty(st.sync.pending) && evictablestorage(st.storage)
evictablestorage(::Union{Region,Reserve,Committed}) = true
evictablestorage(::Union{Evicted,Released}) = false

# The last pressure step in acquire!: evicts idle resources, least recently
# used first (the token of the last access), until their bytes cover `need`,
# and returns the tokens the evicted storage was retired with, for acquire! to
# wait on. Each copy is a submission of its own. Under the record's lock the
# resource becomes Evicted and the plans that name it (all register in its
# dependents, resizing-and-raytracing.jl B.6) are marked. On a device whose
# readback memory is the same heap as its device memory (Apple M5) a host copy
# frees nothing: no eviction.
function evictfor!(pool, need)
    caps(pool.device).separatehostheap || return Token[]
    tokens = Token[]
    freed = 0
    for st in leastrecentlyused(pool)                 # weak references, a snapshot; each rechecked under its lock
        freed >= need && break
        host = acquirehost!(pool, bytes(st))          # host memory, before the lock; nothing evicted if this fails
        retired = nothing
        try
            # The naming plans' sync locks first, then st's (section 8): runs
            # lock only their inbox, so this is what keeps a run from starting
            # its submission while st is evicted under it. The eviction is an
            # eager access: it puts st into those plans' inboxes, so a run
            # that already looked locks st, finds it Evicted and restores it.
            retired = lockresources(((st, Read()),); lent = true) do recs, x
                evictable(st) || return nothing
                # Nothing is pending on st (evictable), so prepare! allocates
                # nothing for it: no acquire! under this record lock.
                copy = submitnow!(((st, Read()),)) do b, after   # waits for the last write; a later write waits for it
                    emitcopy!(b, nothing, copyof(after(st) => host), Deps((), ()))
                end
                last = lasttokens(st.sync, st)      # the accesses in flight, the naming plans' last runs, the copy
                retire!(pool, st.storage, last)
                st.storage = Evicted(host, copy)
                foreach(markrestore!, keys(st.dependents))   # @atomic plan.restore = true, through the weak reference
                last
            end
        finally
            retired === nothing && release!(pool, host)   # not evictable after all, or a throw
        end
        retired === nothing || (append!(tokens, retired); freed += bytes(st))
    end
    return tokens
end

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
# resource, so it waits for the eviction copy (recorded as a read); the host
# copy is retired with the submission's token. Storage missing or too small
# (evicted after the submitter looked, or resized meanwhile): Again, nothing
# emitted. No eviction can happen while the locks are held.
restoresteps(st, ::Union{Region,Reserve,Committed}, _) = ()   # not evicted, or restored meanwhile
restoresteps(st, ::Evicted, ::Missing) = Again()
restoresteps(st, ev::Evicted, new) = sizeof(new) >= bytes(st) ?        # bytes(st): the requested size
    (Restore(st, ev.host, new, min(sizeof(ev.host), bytes(st))), CellWrite(st, address(new), st.dims)) : Again()
isevicted(::Evicted) = true
isevicted(_) = false
emitstep!(b, s::Restore) = emitcopy!(b, nothing, copyof(s.hostcopy => s.new, s.bytes), Deps((), ()))

# Everything else that meets an evicted array, by dispatch on the storage:
# - an eager call, a store, a resize!, becoming a composite's member: the
#   submission that applies them restores it first (above); the calls only
#   enqueue;
# - a pending resize!: the restore allocates the requested size and copies
#   min(host copy, requested) bytes, so the Resize after it is in place;
# - a host read: waits for the eviction token and copies from the host copy
#   under the record's lock, no submission;
# - an image: restored into a new image under a new, unused table index, like
#   a regrown texture (resizing-and-raytracing.jl B.9): commands submitted
#   before the eviction may still read the old index;
# - the last hold: the host copy is retired after the eviction token (section 10).
# The pool's least-recently-used list holds weak references, so it keeps
# nothing alive; a released state leaves it (section 10).

# The budget check and the new block happen under one lock, so two threads
# cannot both pass it. rawalloc is a driver call, not a GPU wait.
tryacquire!(pool, kind, bytes, constraint) = lock(pool.lock) do
    r = suballocate!(pool, kind, bytes, constraint)
    r === nothing || return r
    used(pool) + blocksize(bytes) > budget(pool.device) && return nothing
    addblock!(pool, rawalloc(pool.device, kind, blocksize(bytes), constraint))   # device-type verb
    suballocate!(pool, kind, bytes, constraint)
end
# budget(dev): the device's budget minus other processes' usage. One meaning.

# Retired with the tokens of its last uses; released once all have passed.
retire!(pool, x, tokens) = push!(pool.inbox, Retired(x, tokens))   # lock-free: safe in a finalizer

function sweep!(pool)
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

# Sparse binding is a queue operation (vkQueueBindSparse, Metal's sparse
# mapping update, cuMemMap), not a recorded command. Core claims a token on the
# sparse channel and records the access under the channel lock; the device
# type's verb binds or unbinds and signals the token. Callers hold the
# records' locks.
function sparsesubmit!(ch, verb!, ranges, accesses)
    waits = waitsfor(accesses)
    lock(ch.lock) do
        token = Token(ch, nexttoken!(ch))
        recordeager!(accesses, token)            # binding and unbinding are not runs: recorded on the SyncState
        verb!(ch, token.value, ranges, waits)    # bindpages! or unbindpages!
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
# A composite's hold (a TLAS or BLAS range, an array GPURef) is counted twice:
# in holds, and in composites, which keeps the array from being evicted
# (section 9).
# Taken under the array's record lock (holdmember!, resizing-and-raytracing.jl
# B.8b), so evictable, checked under the same lock, sees both or neither.
addcompositehold!(st::ArrayState) = (addhold!(st); Threads.atomic_add!(st.composites, 1))
dropcompositehold!(st::ArrayState) = (Threads.atomic_sub!(st.composites, 1); drophold!(st))
addcompositehold!(x) = addhold!(x)      # a BLAS or an array GPURef held by a range: never evicted anyway
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
usable(r::ArrayBox) = r.sync.freed && throw(ArgumentError("use of a GPURef after free!"))

# Every resource a top-level node names, wherever it sits in an argument
# (inside a Broadcasted, a keyword, a view's parent, a flag, a loop body):
# one hold per graph, counted per node so delete! knows when the last node
# naming it went.
function name!(g::Graph, n::Node)
    for st in unique(persistentresources(n))
        k = get(g.named, st, 0)
        k == 0 && addhold!(st)
        g.named[st] = k + 1
    end
end

# free! of a root array drops the creator's hold once, whatever the number of
# threads; the state's finalizer does the same through the inbox.
free!(a::MantleArray) = free!(a, a.role)
free!(a, ::Derived) = throw(ArgumentError("free! of a view; free its parent"))
free!(a, ::Root) = dropcreator!(a.state)
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
    unregister!(g.plan)                              # dependents and value GPURefs drop their weak entries
    retire!(pool(g.device), g.plan, lasttokens(g.plan))   # argument block, entry, recordings, descriptors
end
# Finalizers only push onto the lock-free inbox; settle! (in sweep!) does the
# work, one item at a time:
#   finalizer(st -> push!(pool(st).inbox, DropCreator(st)), state)   # settle!: dropcreator!(st)
#   finalizer(g -> push!(pool(g.device).inbox, Release(g)), g)       # settle!: (@atomicswap g.freed = true) || release!(g)
#   finalizer(x -> push!(pool(x).inbox, DropHolds(x)), x)            # an ArrayBox, TLAS, BLAS or RenderObject:
#                                                                    #   settle! drops the composite holds it has
# The state's finalizer runs only when no array, view, graph or composite
# refers to it (g.named and the composites' members keep it alive), so it
# drops the creator's hold if free! did not; other holders drop theirs through
# their own finalizers. A finalized graph cannot be running (run! holds a
# reference), so its release needs no graph lock. `settle!` takes records
# without the lent check and never waits on a run.

# ══ 11. Eager calls: the same dispatch!, submitted now ════════════════════════

# A GPURef argument of an eager call is packed with its value at the call.
# Conditions exist in graphs only (when! takes a Graph): an eager call runs.
# Its arguments' pending changes, and the restore of an evicted one, are
# applied in front of the launch by the same submission (submitapplying).
function dispatch!(dev::Device, kernel, args::Tuple, ndrange; origin = nothing)
    k = kernel!(dev, kernel, argtypes(args))
    ch = channel(dev, Main())
    packed = acquire!(pool(dev), Unified(), argumentbytes(k))   # fallible: before any lock
    token = nothing
    try
        token = submitapplying(uses(k, args); ch) do after   # under the locks, after the pending changes' steps
            foreach(usable, resources(args))         # freed through the creator's handle: throws, under the lock
            packarguments!(packed, k, args, after)   # each array's storage after the pending changes, its requested dims
            ext = launchextents(dev, k, resolve(ndrange, args))   # requested lengths; a kernel-written length launches indirect
            b = Builder(dev, ch)
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
# An array in Readback memory (a graph's output): its pending changes first (a
# submission of their own only if there are any), then a wait for every write
# to it, eager or by a run (waitsfor: the runs' last tokens are read under the
# naming plans' sync locks; the wait happens outside the locks), and a memcpy.
function Base.Array(a::MantleArray, ::Readback)
    usable(a)
    haspending(a) && submitnow!((b, after) -> nothing, ((a.state, Read()),))
    waitfor(lockresources(((a.state, Read()),)) do recs, x
        waitsfor(((a.state, Read()),))
    end)
    return copy(hostview(a))
end
function Base.Array(a::MantleArray{T,N}, _) where {T,N}
    usable(a)                                        # a host read through a freed handle throws, as other calls do
    bytes = readbackbytes(a)                         # size(a), or the capacity for a device-written length
    staging = acquire!(pool(a), Readback(), bytes + cellbytes(a))
    token = nothing
    try
        token = submitnow!(((a.state, Read()),)) do b, after   # the pending changes first, then the copy
            emitcopy!(b, nothing, copyof(region(after(a.state), a) => staging, bytes), Deps((), ()))
            foreach(c -> emitcopy!(b, nothing, c, Deps((), ())), cellcopies(a, staging))   # the cell, for a
        end                                          #   kernel-written length; none for a length the host knows
        waitfor(token)
        dims = readdims(a, staging)                  # from the copied cell, or size(a)
        return copyto!(Array{T,N}(undef, dims), hostview(staging, T, dims))
    finally
        retire!(pool(a), staging, token === nothing ? () : (token,))
    end
end
# An evicted array: Array(a, kind, ::Evicted) waits for the eviction token and
# copies from the host copy, holding the record's lock (section 9).

# `size(a)` is the host view. The size a kernel wrote is read with
# devicesize(a) (name open): a copy of the cell in a submission of its own, a wait, a memcpy.

# ══ 12. Device-type verbs (examples of the only code a device type writes) ════

# Vulkan: the barrier core derived, then the launch. The one barrier call site.
# A launch inside a when! block is indirect like any other; the flag was
# multiplied into its command by the head or a prepare node (section 2).
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

# Called by core's submit! under the channel lock. Present needs binary
# semaphores: the segment writing the image waits on the acquire semaphore and
# signals the image's rendered semaphore, which the present waits on.
function submitnative!(ch::VulkanChannel, value, parts, waits, signals, present)
    vkQueueSubmit2(ch.queue, submitinfo(parts;
        wait = [timelinewaits(waits); binarywaits(present)],   # waits: Tokens, one per channel
        signal = [(ch.timeline, value); signals; binarysignals(present)]))
    present === nothing || vkQueuePresentKHR(ch.queue, presentinfo(present))
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
    opened::Dict{Any,Device}        # hardware key => the device object
    instance::Ref{Any}
end
const REGISTRY = DeviceRegistry(ReentrantLock(), Type[], Dict{Any,Device}(), Ref{Any}(nothing))
# livegraphs(REGISTRY): the graphs of every opened device, held weakly by the
# device (for inhostfunction, resizing-and-raytracing.jl B.3).
# Mantle's __init__: push!(REGISTRY.types, LavaDevice). MantleCUDAExt's: CudaDevice.

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
# current address.
function withview(f, ::Type{CuArray}, a::MantleArray{T,N,CudaDevice}, stream) where {T,N}   # (name open)
    st = a.state
    ch = foreignchannel(stream)                          # before the hold: nothing to undo if it throws
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
                    s = needsapplying(st) ? submitwith!(channel(device(a), Main()), recs, prepared, (), OwnWork(after -> [])) :
                                            nothing      # nothing pending, not evicted: no submission
                    s === nothing || done(s) || return s # Again or WaitFor: nothing submitted
                    ready = s === nothing ? Token[] : [first(s)]
                    streamwaits!(stream, [ready; waitsfor(((st, Write()),))])   # the stream waits for Mantle's accesses
                    w = unsafe_wrap(CuArray, CuPtr{T}(address(st.storage)), size(a); own = false)   # applied storage
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
                t = lock(ch.lock) do
                    t = Token(ch, nexttoken!(ch))
                    recordeager!(((st, Write()),), t)    # Mantle's later uses wait for the block's work; tells the plans
                    signal!(stream, t.value)             # written by the stream after the block's work
                    t
                end
                st.sync.lent = false
                t
            end
        end
    finally
        token === nothing ? drophold!(st) : dropafter!(st, (token,))
    end
end
