# A readback is declared in the graph. Waiting for its result never schedules
# device work. The staging Buffer is retained by a snapshot after plan teardown.
mutable struct ReadbackSnapshot
    dev::Any
    buffer::Any
    token::Any
    data::Any
    lock::ReentrantLock
end

"""A one-shot CPU result of a graph execution; use `isready`, `wait`, or `fetch`."""
mutable struct ReadbackFuture{T,N}
    event::Base.Event
    @atomic assigned::Bool
    snapshot::Union{Nothing,ReadbackSnapshot}
    failure::Any
end
ReadbackFuture{T,N}() where {T,N} =
    ReadbackFuture{T,N}(Base.Event(), false, nothing, nothing)

mutable struct GraphReadback{T,N} <: Resource
    source::Any
    dims::NTuple{N,Int}
    buffer::Any
    owner::WeakRef
    pending::Vector{ReadbackFuture{T,N}}
    active::Vector{ReadbackFuture{T,N}}
    previous::Union{Nothing,ReadbackSnapshot}
    recordedsource::Any
    lock::ReentrantLock
end
Base.size(r::GraphReadback) = r.dims
Base.length(r::GraphReadback) = prod(r.dims)
Base.eltype(::GraphReadback{T}) where {T} = T
storage(r::GraphReadback) = storage(r.buffer)

readbackdims(src) = size(src)
readbackdims(src::Buffer{<:Any,1}) = (length(src),)

readbackgraph(t::TransientBuffer) = t.graph
readbackgraph(t::ResourceView) = readbackgraph(t.parent)
readbackgraph(t) = nothing
readbackdevice(t::TransientBuffer) = t.graph.dev
readbackdevice(t::ResourceView) = readbackdevice(t.parent)
readbackdevice(t::Buffer) = t.dev
readbackdevice(t) = Device(KA.get_backend(t))

"""
    Readback(array) -> ReadbackFuture
    Readback(graph, array) -> ReadbackFuture

Declare a CPU snapshot at this point in the graph. The one-argument form accepts
an array owned by a graph (`Transient.Buffer` or its `viewof` view). For external
buffers, name the graph explicitly. No GPU work runs here.

Declare outputs before `Plan`, then `record!` and `run!` the plan. `fetch(future)`
waits for that execution and returns an owning CPU `Array`, without submitting
work. Use `Plan(g; budget=Inf)` to keep the run in one GPU submission.

After compilation, calling `Readback` again for an existing output requests its
next execution, reusing the recorded transfer. Each future belongs to one execution;
previous results survive replay and `free!(plan)`. Before replay overwrites an
unfetched snapshot, Mantle waits and copies it to CPU memory. Thus readbacks can
introduce host backpressure between runs, but never extra GPU submissions.

Outputs have fixed shapes. Declare them outside `repeat!`, after the loop.
Replays reject a changed source shape; multiple snapshots of the same source
can be fetched individually, but requesting that ambiguous source again throws.
`Readback()` without an array remains the allocator's readback memory kind.
"""
function Readback(t::Union{TransientBuffer,ResourceView})
    g = readbackgraph(t)
    g isa Graph || throw(ArgumentError("Readback: this array has no owning graph; use Readback(graph, array)"))
    return Readback(g, t)
end

function Readback(g::Graph, src)
    src isa Union{TransientBuffer,ResourceView,Buffer} || isdevicearray(src) ||
        throw(ArgumentError("Readback: expected a dense graph or device array"))
    if src isa Union{TransientBuffer,ResourceView}
        root = src isa ResourceView ? src.parent : src
        root isa TransientBuffer && root.graph !== g &&
            throw(ArgumentError("Readback: array belongs to another graph"))
    end
    readbackdevice(src) === g.dev ||
        throw(ArgumentError("Readback: source and graph belong to different devices"))
    dims = readbackdims(src)
    checkedbytes(eltype(src), dims) > 0 || throw(ArgumentError("Readback: empty output"))
    r = if g.readbacksealed
        matches = filter(r -> samereadback(r.source, src), g.readbacks)
        isempty(matches) && throw(ArgumentError("Readback: declare outputs before compiling the graph"))
        length(matches) == 1 || throw(ArgumentError(
            "Readback: source has multiple snapshots; use a distinct output buffer for replay"))
        only(matches)
    else
        # Separate declarations can capture the same source before and after a
        # later write. The read dependency makes that snapshot part of scheduling.
        r = GraphReadback{eltype(src),length(dims)}(src, dims, nothing, WeakRef(nothing),
            ReadbackFuture{eltype(src),length(dims)}[], ReadbackFuture{eltype(src),length(dims)}[],
            nothing, nothing, ReentrantLock())
        readbackpass!(g, r)
        push!(g.readbacks, r)
        r
    end
    return lock(r.lock) do
        if g.readbacksealed
            pl = r.owner.value
            (pl === nothing || (@atomic pl.freed)) &&
                throw(ArgumentError("Readback: the output's plan has been freed"))
        end
        f = ReadbackFuture{eltype(src),length(dims)}()
        push!(r.pending, f)
        f
    end
end
samereadback(a, b) = a === b
samereadback(a::ResourceView, b::ResourceView) =
    a.parent === b.parent && a.dims == b.dims && a.offset == b.offset

# A shared-memory backend uses an ordinary recorded copy kernel. Vulkan lowers
# the same declaration to a native transfer into host-cached staging instead.
@kernel function graph_readback_copy!(dst, src)
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i]
end
# A native transfer is a device capability; graph declaration, access direction
# and lifetime stay in core. Shared-memory backends use the ordinary copy kernel.
nativebuffercopy(::Device) = false
function readbackbinding end
readbacksource(g, src) = src
readbacksource(g, src::ResourceView) = slice(g, src.parent, elementrange(src))
function readbackpass!(g::Graph, r::GraphReadback)
    if nativebuffercopy(g.dev)
        p = Pass("Readback", :readback)
        push!(p.targets, r.source)
        p.dst = r
        src = readbacksource(g, r.source)
        push!(p.usages, resourceid(g, src) => CopySrc)
        push!(p.usages, resourceid(g, r) => CopyDst)
        push!(g.passes, p)
        touch!(g, rootresource(src))
        return p
    end
    p = Pass("Readback", :compute)
    push!(p.dispatches, Dispatch(graph_readback_copy!, (r, r.source), length(r), nothing))
    src = readbacksource(g, r.source)
    push!(p.usages, resourceid(g, src) => Storage{BufferKind,ReadOnly})
    push!(p.usages, resourceid(g, r) => Storage{BufferKind,WriteOnly})
    push!(g.passes, p)
    touch!(g, rootresource(src))
    return p
end

function assignreadback!(f::ReadbackFuture; snapshot = nothing, failure = nothing)
    f.snapshot, f.failure = snapshot, failure
    @atomic f.assigned = true
    notify(f.event)
    return nothing
end

function materializereadback!(s::ReadbackSnapshot)
    lock(s.lock) do
        s.data === nothing || return s.data
        waitfor!(s.dev, s.token)
        # An owning copy, never a borrowed pointer into the next run's staging.
        s.data = copy(hostview(s.dev, s.buffer.store))
        s.buffer = nothing
        return s.data
    end
end
function Base.isready(f::ReadbackFuture)
    (@atomic f.assigned) || return false
    f.failure === nothing || return true
    s = f.snapshot
    # `data` and `buffer` are guarded even when another task is fetching.
    return lock(s.lock) do
        s.data !== nothing || passed(s.dev, s.token)
    end
end
function Base.wait(f::ReadbackFuture)
    wait(f.event)
    f.failure === nothing || throw(f.failure)
    materializereadback!(f.snapshot)
    return f
end
function Base.fetch(f::ReadbackFuture{T,N}) where {T,N}
    wait(f)
    return f.snapshot.data::Array{T,N}
end

function bindreadbacks!(pl::Plan)
    g = pl.graph
    isempty(g.readbacks) && (g.readbacksealed = true; return nothing)
    # A graph's placed transients and readback addresses cannot serve two live
    # plans. Check all owners before changing anything.
    for r in g.readbacks
        old = r.owner.value
        (old === nothing || (@atomic old.freed)) ||
            throw(ArgumentError("Plan: graph readbacks already belong to a live plan"))
    end
    try
        for r in g.readbacks
            r.owner = WeakRef(pl)
        end
        ids = Set(resourceid(g, r) for r in g.readbacks)
        any(p -> p.loop !== nothing && any(u -> first(u) in ids, p.usages), g.passes) &&
            throw(ArgumentError("Readback: declare snapshots after repeat!, outside its body"))
        for r in g.readbacks
            a = allocate(pool(g.dev), g.dev, Readback(), eltype(r), size(r); blocksize=4 << 20)
            r.buffer = Buffer{eltype(r),length(size(r))}(a, length(r), length(r), g.dev)
        end
    catch err
        freereadbacks!(pl, err)
        rethrow()
    end
    g.readbacksealed = true
    return nothing
end

function preparereadbacks!(pl::Plan)
    for r in pl.graph.readbacks
        lock(r.lock) do
            # Preserve a previous, unfetched result before the same recording
            # writes this staging again. Waiting here schedules no device work.
            r.previous === nothing || materializereadback!(r.previous)
            r.previous = nothing
            readbackdims(r.source) == r.dims || throw(DimensionMismatch(
                "Readback: source shape changed; declare a new graph for the new shape"))
            # Native copies bake buffer handles, not patchable shader pointers.
            # A source move therefore needs fresh compiled bindings and commands.
            if r.recordedsource !== nothing &&
               r.recordedsource != readbackbinding(pl.graph.dev, r.source)
                invalidate!(pl)
                pl.replaced = true
            end
            append!(r.active, r.pending)
            empty!(r.pending)
        end
    end
    return nothing
end
function commitreadbacks!(pl::Plan, token)
    for r in pl.graph.readbacks
        lock(r.lock) do
            isempty(r.active) && return
            s = ReadbackSnapshot(pl.graph.dev, r.buffer, token, nothing, ReentrantLock())
            r.previous = s
            for f in r.active
                assignreadback!(f; snapshot=s)
            end
            empty!(r.active)
        end
    end
    return nothing
end
function failreadbacks!(pl::Plan, err)
    for r in pl.graph.readbacks
        lock(r.lock) do
            for f in Iterators.flatten((r.active, r.pending))
                assignreadback!(f; failure=err)
            end
            empty!(r.active)
            empty!(r.pending)
        end
    end
    return nothing
end
function freereadbacks!(pl::Plan, err = ArgumentError("readback plan was freed before execution"))
    for r in pl.graph.readbacks
        lock(r.lock) do
            r.owner.value === pl || return
            for f in Iterators.flatten((r.active, r.pending))
                assignreadback!(f; failure=err)
            end
            empty!(r.active)
            empty!(r.pending)
            # Submitted futures retain their buffer and token independently.
            r.previous = nothing
            r.buffer = nothing
            r.owner = WeakRef(nothing)
            r.recordedsource = nothing
        end
    end
    return nothing
end
