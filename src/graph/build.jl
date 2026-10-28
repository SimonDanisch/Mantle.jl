# Graph-building verbs. They name the concrete `Graph` and `Pass` from
# `graph/types.jl`, so they cannot live in `runtime/api.jl`, which loads first.

"""
    dispatch!(graph, kernel, args, ndrange; group = nothing, name = nothing)

One kernel launch as its own pass, ordered by what the kernel does to `args`,
which is inferred from the kernel body ([`kerneltouches`](@ref)). `name` defaults
to the kernel's name. `group` is the workgroup size; the default
([`defaultgroup`](@ref)) makes a 2-D workgroup one row of 64, so image-shaped
passes should give one. `ndrange` may be a [`DeviceRange`](@ref); the dispatch is
then ordered after whatever wrote the count.
"""
function dispatch!(g::Graph, kernel, args::Tuple, ndrange;
                   group = nothing, name = nothing)
    refuserefs(args)
    refuseempty(kernel, ndrange)
    p = newpass(g, name === nothing ? passname(kernel, args) : String(name), :compute)
    push!(passes(g), p)
    h = handle(g, p)
    declare!(h, args, refusenothing(kernel, args,
                 argument_usage(g.dev, kernel, args, ndrange, group)))
    n = countresource(ndrange)
    n === nothing || indirectcount!(h, n)
    push!(dispatches(p), Dispatch(kernel, args, ndrange, group))
    return p
end

"""
    refusenothing(kernel, args, touches) -> touches

Throw if `touches` marks no argument as touched. That means the access walk missed
a memory access (typically an intrinsic without `intrinsic_usage`), and a missed
store would silently remove a barrier.
"""
function refusenothing(@nospecialize(kernel), args::Tuple, touches)
    any(touched, touches) && return touches
    throw(ArgumentError(
        "dispatch!: `$kernel` is declared to touch none of its $(length(args)) " *
        "arguments, which cannot be right for a kernel worth launching. Either " *
        "it reads or writes something the access walk cannot see -- an intrinsic " *
        "with no `intrinsic_usage`, most likely -- or the dispatch has no reason " *
        "to exist."))
end

"""
    refuseempty(kernel, ndrange)

Throw for an empty host ndrange, which would otherwise fail as a `DivideError` in
`KernelAbstractions.partition`. A [`DeviceRange`](@ref) may be zero at run time
and passes.
"""
function refuseempty(@nospecialize(kernel), ndrange)
    ndrange isa DeviceRange && return nothing
    n = ndrange isa Tuple ? prod(ndrange) : ndrange
    n isa Integer && n <= 0 && throw(ArgumentError(
        "dispatch!: `$kernel` was given an ndrange of $ndrange, which is empty. " *
        "A result with no elements needs no pass, so skip the dispatch where " *
        "the shape is decided. A count only the device knows is a `DeviceRange`."))
    return nothing
end

"""
    dispatch!(graph, call, args; name = nothing)

Declare a call that launches its own work (e.g. `dispatch!(g, mul!, (C, A, B))`),
ordered by what it reads and writes. Its accesses come from an
[`argument_usage`](@ref) method declared next to the function; without one,
[`UndeclaredCall`](@ref) is thrown. `runscalls(device)` says which backends can
run calls.
"""
function dispatch!(g::Graph, call, args::Tuple; name = nothing)
    refuserefs(args)
    p = newpass(g, name === nothing ? string(nameof(call)) : String(name), :compute)
    push!(passes(g), p)
    h = handle(g, p)
    declare!(h, args, argument_usage(g.dev, call, args, nothing, nothing))
    push!(dispatches(p), Dispatch(call, args, nothing, nothing))
    return p
end

"""
    passname(kernel, args) -> String

The kernel's name, or the first argument's name when that is a function, so a
higher-order kernel (a work-queue mapper) is named after the kernel it runs.
"""
passname(kernel, args::Tuple) =
    (!isempty(args) && first(args) isa Function) ? string(nameof(first(args))) :
    string(nameof(kernel))

"""
    declare!(pass, args, touches)

Register one usage per device buffer reachable from each argument, because a
barrier is scoped to a buffer and arguments may be containers. An argument's
`Touch` applies to all of its leaves.
"""
function declare!(h, args::Tuple, touches::Vector{Touch})
    g, p = graphof(h), passof(h)
    leaves = Any[]
    for (a, t) in zip(args, touches)
        touched(t) || continue
        empty!(leaves)
        resourceleaves!(leaves, g, a)
        for r in leaves
            push!(p.usages, resourceid(g, r) => usagetype(t, resourcekind(r)))
            # Liveness tracks the parent: a slice's interval is its buffer's.
            touch!(g, rootresource(r))
        end
    end
    return p
end

"""
    resourceleaves!(acc, graph, x) -> acc

Append every device buffer reachable from `x` to `acc`. Resources and backend
arrays are leaves; other values are walked field by field. A `ResourceView`
yields the slice of its parent it covers, interned per graph so repeated views
are one resource. Native array parents conservatively name their whole storage
owner: scalar/vector views can use different element widths over those bytes.
"""
resourceleaves!(acc, g, v::ResourceView, seen::Base.IdSet{Any}, depth::Int) =
    (push!(acc, isdevicearray(v.parent) ? v.parent : slice(g, v.parent, elementrange(v))); acc)

resourceleaves!(acc, g, x::Resource, seen::Base.IdSet{Any}, depth::Int) = (push!(acc, x); acc)
resourceleaves!(acc, g, x::BufferRange, seen::Base.IdSet{Any}, depth::Int) =
    (push!(acc,isdevicearray(x.parent) ? x.parent : x); acc)
resourceleaves!(acc, g, @nospecialize(x)) = resourceleaves!(acc, g, x, Base.IdSet{Any}(), 0)

# `Attribute(p, x)` already declared it as `Vertices`; declaring it again would
# add a storage read the vertex fetch does not do.
resourceleaves!(acc, g, ::Attr, seen::Base.IdSet{Any}, depth::Int) = acc

# Device objects reach their pool and device, so the walk needs both a cycle
# guard (`seen`) and a depth limit.
const LEAFDEPTH = 16

function resourceleaves!(acc, g, @nospecialize(x), seen::Base.IdSet{Any}, depth::Int)
    isdevicearray(x) && return (push!(acc, x); acc)
    depth > LEAFDEPTH && return acc
    T = typeof(x)
    carries(T) || return acc
    isstructtype(T) || return acc
    if ismutabletype(T)
        x in seen && return acc
        push!(seen, x)
    end
    for i in 1:fieldcount(T)
        isdefined(x, i) || continue
        resourceleaves!(acc, g, getfield(x, i), seen, depth + 1)
    end
    return acc
end

"""
    vertextouches(device, shader, args) -> Vector{Touch}
    fragmenttouches(device, shader, args) -> Vector{Touch}

What each draw stage does to its arguments: the vertex stage to `args`, the
fragment stage to `frag_args` (its function also takes the varyings as a leading
argument). Backends answer like [`kerneltouches`](@ref), from what they compile.
"""
function vertextouches end
function fragmenttouches end

"""
    isdevicearray(x) -> Bool

Whether `x` is a backend array, i.e. a leaf for [`resourceleaves!`](@ref).
Backends answer for their own array type.
"""
isdevicearray(@nospecialize(x)) = false

"""
    denseoperands(xs...) -> Bool

Whether every operand is absent, a graph resource a library call can be handed,
or the backend's own dense device array. Shape and type do not suffice: packed
weight wrappers expose a matrix shape over encoded `UInt32` words.
"""
denseoperands(xs...) = all(x -> x === nothing || isdevicearray(x) ||
                               x isa Union{Buffer,TransientBuffer,ResourceView}, xs)

"""
    repeat!(f, graph, n) -> passes
    repeat!(f, graph, maxiters, count) -> passes
    repeat!(f, graph, maxiters; while_nonzero = flag) -> passes
    repeat!(graph, kernel, args, ndrange; max, count, while_nonzero, group, name) -> passes

A loop whose body `f(i)` is declared and compiled once and submitted `n` (or
`maxiters`) times. `i` is a one-element device `Int32` buffer holding the
1-based iteration. An optional device-side gate, re-read before each iteration,
lets the body end the loop:

  * `count`: a device integer; iteration `i` runs while `i <= count[]`.
  * `while_nonzero`: a device integer; the iteration runs while it is nonzero.

Gated-off iterations are discarded at execution but still cost their barriers,
gate dispatch and predicate test, so `maxiters` should be a tight bound. Each
iteration starts behind a full barrier. A transient the body reads before
writing, or shares with surrounding passes, lives across the whole loop; others
are scoped to one iteration. The body may not touch an image. Loops may nest:
each child index resets for every parent iteration, and child gates include all
enclosing device predicates. `maxiters = 1` is a plain gated region without
these limits. Iteration boundaries are submission cut points (see
`partitionranges`). The kernel form uses one `dispatch!` as the body.
"""
function repeat!(g::Graph, kernel, args::Tuple, ndrange;
                 max::Integer, count = nothing, while_nonzero = nothing,
                 group = nothing, name = nothing)
    repeat!(g, max, count; while_nonzero) do _
        dispatch!(g, kernel, args, ndrange; group, name)
    end
end

function repeat!(f, g::Graph, maxiters::Integer, count = nothing;
                 while_nonzero = nothing)
    maxiters >= 1 || throw(ArgumentError("repeat!: maxiters must be at least 1, got $maxiters"))
    (count === nothing || while_nonzero === nothing) || throw(ArgumentError(
        "repeat!: give at most one gate — a `count` positionally, or " *
        "`while_nonzero = flag`."))
    gated = count !== nothing || while_nonzero !== nothing
    # A single iteration is a plain gated region: no index, no head barrier, no
    # recording cut, and none of the loop restrictions.
    looped = maxiters > 1
    index = Buffer(g.dev, Int32[1])
    # Outside the body: resets the index to 1 on every run.
    looped && gate!(g, loop_start!, (index,), "repeat!/start")
    firstloop = length(passes(g)) + 1
    # The gate is the body's first pass and is not itself gated: it writes the
    # flag the rest of the body is gated on.
    pred = gated ? Buffer(g.dev, [Predicate(0)]) : nothing
    if count !== nothing
        gate!(g, gate_count!, (pred, count, index, ()), "repeat!/gate")
    elseif while_nonzero !== nothing
        gate!(g, gate_nonzero!, (pred, while_nonzero, ()), "repeat!/gate")
    end
    firstbody = length(passes(g)) + 1
    f(index)
    for k in firstbody:length(passes(g))
        pp = passes(g)[k]
        if looped
            for (id, _) in pp.usages
                resourcekind(g.ids.by_id[id]) isa ImageKind && throw(ArgumentError(
                    "repeat!: pass \"$(pp.name)\" touches an image. A loop body runs " *
                    "again behind itself, so an image's layout would have to be the " *
                    "same on the way in and out of every iteration; images in a " *
                    "`repeat!` body are not supported."))
            end
        end
        gated || continue
        # An inner gate must run even when its enclosing body is discarded:
        # it writes the conjunction, clearing a flag left set by a previous run.
        inheritgate!(g, pp, pred) && continue
        if pp.predicate !== nothing
            # Its own gate now includes this parent's predicate.
            continue
        end
        pp.predicate = (pred, 0)
        # Device-sized dispatches are discarded on every backend because the
        # prepare writes zero groups (`emitprepares!`); fixed-size work needs
        # backend predication. The message names `typeof(dev)` because showing
        # a device prints its whole pool.
        supportspredicate(g.dev) || devicesized(pp) || throw(ArgumentError(
            "repeat!: pass \"$(pp.name)\" has fixed-size work and " *
            "$(nameof(typeof(g.dev))) cannot discard recorded work on a " *
            "device-written predicate. Size every dispatch of a gated pass " *
            "on the device (a `DeviceRange`), or loop on the host."))
        # Declaring the predicate read puts a barrier after the gate's write.
        push!(pp.usages, resourceid(g, pred) => Predicated)
        touch!(g, pred)
    end
    body = passes(g)[firstloop:end]
    looped || return body
    # The body's last pass: advances the index for the next iteration.
    gate!(g, loop_advance!, (index,), "repeat!/advance")
    body = passes(g)[firstloop:end]
    loop = Loop(Int(maxiters), index)
    for pp in body
        if pp.loop === nothing
            pp.loop = loop
        elseif !withinloop(pp.loop, loop)
            childloop(pp.loop, nothing).parent = loop
        end
    end
    return body
end

# Keep a pass's one hardware predicate. Nested gates combine their parents on
# the device rather than nesting backend-specific conditional-rendering scopes.
function inheritgate!(g::Graph, p::Pass, parent)
    length(p.dispatches) == 1 || return false
    d = only(p.dispatches)
    d isa Dispatch || return false
    (d.kernel === gate_count! || d.kernel === gate_nonzero!) || return false
    args = (Base.front(d.args)..., (last(d.args)..., parent))
    p.dispatches[1] = Dispatch(d.kernel, args, d.ndrange, d.group)
    empty!(p.usages)
    declare!(handle(g, p), args, argument_usage(g.dev, d.kernel, args, d.ndrange, d.group))
    return true
end

"""Before a loop: the index every run starts from."""
function loop_start!(index)
    @inbounds index[1] = Int32(1)
    return nothing
end

"""Increment the loop index; the body's last pass."""
function loop_advance!(index)
    @inbounds index[1] += Int32(1)
    return nothing
end

"""
    Predicate(go::UInt32)

A device-written gate flag: nonzero runs, zero discards (the Vulkan conditional
rendering convention, used on every backend). A distinct element type so the
backend gives the buffer its predicate usage flag, as `DrawIndirectCommand` does
for indirect usage.
"""
struct Predicate
    go::UInt32
end

"""The gate for `repeat!(…, count)`: this iteration runs while `index[] <= count[]`."""
@inline parentsrun(parents::Tuple) = all(p -> (@inbounds p[1].go != 0), parents)

function gate_count!(pred, count, index, parents=())
    @inbounds pred[1] = Predicate(parentsrun(parents) && index[1] <= Int32(count[1]) ? UInt32(1) : UInt32(0))
    return nothing
end

"""The gate for `repeat!(…; while_nonzero)`: this iteration runs while it is set.

Re-read every iteration, so a body that empties the thing it is draining stops
the loop, and one that refills it carries on."""
function gate_nonzero!(pred, flag, parents=())
    @inbounds pred[1] = Predicate(parentsrun(parents) && flag[1] != 0 ? UInt32(1) : UInt32(0))
    return nothing
end

# A single-item loop-control pass: a kernel where the device compiles one, else
# a call, which a host-walking device runs on the host arrays (these touch only
# element 1 and use no intrinsics). The call form refuses devices that can do
# neither.
gate!(g::Graph, f, args::Tuple, name) =
    kisupported(g.dev, f) ? dispatch!(g, f, args, 1; name) : dispatch!(g, f, args; name)

# Needed by the call form, whose body the access walk does not read.
argument_usage(::Type{typeof(gate_count!)}, ::Type{<:Tuple{Any,Any,Any}}) =
    (WRITE, READ, READ)
argument_usage(::Type{typeof(gate_nonzero!)}, ::Type{<:Tuple{Any,Any}}) = (WRITE, READ)
argument_usage(::Type{typeof(gate_count!)}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (WRITE, READ, READ, READ)
argument_usage(::Type{typeof(gate_nonzero!)}, ::Type{<:Tuple{Any,Any,Any}}) = (WRITE, READ, READ)
argument_usage(::Type{typeof(loop_start!)}, ::Type{<:Tuple{Any}}) = (WRITE,)
argument_usage(::Type{typeof(loop_advance!)}, ::Type{<:Tuple{Any}}) = (READ | WRITE,)

"""
    supportspredicate(device) -> Bool

Whether the device can discard fixed-size recorded work (a draw, a host-sized
dispatch) on a predicate read at execution time. Only needed when a gated
[`repeat!`](@ref) pass holds fixed-size work. `true` in core, since a
host-walking backend reads the flag between passes; recording backends override.
"""
supportspredicate(::Any) = true

"""Whether the pass has no draws and only `DeviceRange` dispatches, so a
discarded run executes nothing on any backend."""
devicesized(p::Pass) = isempty(p.draws) && all(d -> d.ndrange isa DeviceRange, p.dispatches)

"""
    resourceid(graph, r) -> Int

The graph's id for a resource, interning it on first sight.
"""
resourceid(g::Graph, r) = resourceid(g.ids, r)


"""
    extrausage(T) -> UInt32

Backend buffer-usage bits a buffer of element type `T` needs beyond the default,
0 in core. Vulkan adds indirect usage for `DrawIndirectCommand` and conditional
rendering for `Predicate`; a missing bit is a validation error at the draw.
"""
extrausage(::Type) = UInt32(0)

Surface(g, win) = (s = WindowSurface(win); push!(g.surfaces, s); s)

# Attachment queries forward to the backend's window type, so a window answers
# them like a `TransientImage` does.
target_view(s::WindowSurface)   = target_view(s.win)

target_image(s::WindowSurface)  = target_image(s.win)

target_extent(s::WindowSurface) = target_extent(s.win)

target_format(s::WindowSurface) = target_format(s.win)

# The pixel type `attachment_format` compiles pipelines for; only the backend
# window can name it.
Base.eltype(s::WindowSurface) = eltype(s.win)

DrawCall(shader, args, count) =
    DrawCall(shader, args, count, (), nothing, nothing, nothing, 1)

# Always `NTuple{4,Float32}`, so `CompiledDraw`'s `V` parameter does not depend on
# the caller's number type. The height keeps its sign: negative puts +y at the top.
asviewport(::Nothing) = nothing
asviewport(v) = NTuple{4,Float32}(v)

Pass(name, kind) = Pass(String(name), kind, Any[], LoadOp[], nothing, nothing, nothing, nothing,
                        DrawCall[], Pair{Int,Type}[], Any[], nothing, nothing)

first_target(p::Pass) = isempty(p.targets) ? p.depth : first(p.targets)

"""
    viewof(parent, dims...; offset = 0) -> ResourceView

A `dims`-shaped view of `parent` starting `offset` elements in. Passing it to a
dispatch declares a slice of the parent covering the view's elements.
"""
viewof(parent, dims::Integer...; offset::Integer = 0) =
    viewof(parent, map(Int, dims); offset)
function viewof(parent, dims::Dims{N}; offset::Integer = 0) where {N}
    # Past the end would silently read other resources' bytes at run time.
    prod(dims) + Int(offset) <= length(parent) || throw(ArgumentError(
        "viewof: a $(dims) view at offset $(offset) covers " *
        "$(prod(dims) + Int(offset)) elements of a parent that has " *
        "$(length(parent)) ($(describe(parent)))"))
    return ResourceView{eltype(parent),N,typeof(parent)}(parent, dims, Int(offset))
end

# A view of a view is flattened onto the same parent with the offsets added, so
# `storage` and `use` never walk a chain.
viewof(parent::ResourceView, dims::Dims{N}; offset::Integer = 0) where {N} =
    viewof(parent.parent, dims; offset = parent.offset + Int(offset))

Base.size(v::ResourceView) = v.dims
Base.size(v::ResourceView, d::Integer) = d <= ndims(v) ? v.dims[d] : 1
Base.ndims(::ResourceView{<:Any,N}) where {N} = N
Base.eltype(::ResourceView{T}) where {T} = T
Base.length(v::ResourceView) = prod(v.dims)
Base.isempty(v::ResourceView) = iszero(length(v))
Base.parent(v::ResourceView) = v.parent
describe(v::ResourceView{T}) where {T} = "view($T, $(v.dims)) of $(describe(v.parent))"

"""The parent elements this view covers; barriers are scoped to this range."""
elementrange(v::ResourceView) = (v.offset + 1):(v.offset + length(v))

# The shaped array over the parent's storage: `GPUArrays.derive` for device
# arrays, a reshaped `view` for host arrays.
storage(v::ResourceView{T}) where {T} = deriveview(T, storage(v.parent), v.dims, v.offset)
deriveview(::Type{T}, a::GPUArrays.AbstractGPUArray, dims::Dims, off::Int) where {T} =
    GPUArrays.derive(T, a, dims, off)
deriveview(::Type{T}, a::AbstractArray{T}, dims::Dims, off::Int) where {T} =
    reshape(view(a, (off + 1):(off + prod(dims))), dims)

Base.length(t::TransientBuffer) = prod(t.dims)
Base.size(t::TransientBuffer) = t.dims
Base.size(t::TransientBuffer, d::Integer) = d <= ndims(t) ? t.dims[d] : 1
# Dense, column-major strides, as `strides(::Buffer)` gives.
Base.strides(t::TransientBuffer) = Base.size_to_strides(1, size(t)...)
Base.ndims(::TransientBuffer{<:Any,N}) where {N} = N
Base.eltype(::TransientBuffer{T}) where {T} = T

Base.isopen(s::WindowSurface) = isopen(s.win)


# The colour a load op clears to, `nothing` unless it is a `Clear`.
clearvalue(::LoadOp) = nothing

clearvalue(c::Clear) = NTuple{4,Float32}(c.value)
clearvalue(c::Clear{NTuple{4,UInt32}}) = c.value

"""The depth clear value, or `nothing` unless the load op is a `Clear`."""
depthclear(::LoadOp) = nothing

depthclear(c::Clear) = Float32(c.value)

"""A transient's name in error messages."""
describe(t::TransientBuffer{T}) where {T} =
    "Transient.Buffer($T, $(ndims(t) == 1 ? only(t.dims) : t.dims))"

count(t::TransientBuffer) = prod(t.dims)

stride(::TransientBuffer) = 1

nbytes(t::TransientBuffer{T}) where {T} = checkedbytes(T,t.dims)

# Asked of the device: backends need different alignments for the same transient.
alignment(c::Compile, t) = alignment(c.graph.dev, t)

arena(::TransientBuffer) = Buffers()

"""
    imageusage(device, T) -> backend usage mask

The backend usage flags (`VkImageUsageFlags`, `MTLTextureUsage`) for an image of
element type `T`. The element type decides, so usage and aspect agree: `Float32`
is a depth attachment on every backend. No core default; each backend answers.
"""
function imageusage end

refit!(::Device, ::TransientResource) = false

"""
    refit!(plan) -> Bool

Resize every tracking transient to its source and recompile if any changed, or
if `remap!` marked the plan `replaced`. `true` means new offsets, arenas and
argument memory; the recording is dropped and the next `run!` records again.
Address-only moves are patched instead (`notify_move!`). `run!` calls this once
per frame after acquiring the image and before recording, so a failure can hand
the image back untouched (`abandonframe!`).
"""
function refit!(pl::Plan)
    # Set by `remap!` on a backend that cannot patch.
    moved = pl.replaced
    pl.replaced = false
    # `|=`, not `any`, so every transient is refitted. The dynamic dispatch over
    # `Vector{TransientResource}` boxes its result (96 bytes per transient), the
    # only allocation in a recorded `run!`.
    for t in pl.graph.transients
        moved |= refit!(pl.graph.dev, t)
    end
    moved || return false
    # The recording names the old placement; `run!` re-records before submitting.
    invalidate!(pl)
    # The old argument layout is retired, not released: frames in flight may
    # still read it, and `reclaim!` frees it after they complete.
    let am = pl.args
        am === nothing || retire!(pool(pl.graph.dev), pl.graph.dev, am.store)
    end
    install!(pl, compile!(Compile(pl)))
    return true
end

"""
    install!(plan, compile) -> plan

Take a compile's passes, placement, loops and argument layout into `plan`,
keeping the measured cost of every pass that survives (costs are keyed by pass:
a recompile may reorder them), and register `plan` with its arenas (idempotent).
"""
function install!(pl::Plan, c::Compile)
    cost = IdDict{Any,Float64}(pp.pass => pl.passcost[i] for (i, pp) in enumerate(pl.passes))
    pl.passes, pl.pipelines = c.passes, c.pipelines
    pl.passcost = [get(cost, pp.pass, NaN) for pp in pl.passes]
    pl.loops = loopspans([pp.pass for pp in pl.passes])
    a = analysis(c)
    pl.slabs, pl.arenas, pl.offsets, pl.parts = a.regions, a.arenas, a.offsets, a.parts
    pl.peak, pl.naive = a.peak, a.naive
    # So a later plan growing the arena patches or invalidates this one.
    foreach(ar -> tenant!(pool(pl.graph.dev), ar, pl), a.arenas)
    pl.args = makeargmemory(pl.graph.dev, c.passes)
    return pl
end

Base.size(t::TransientImage) = (t.width, t.height)
Base.eltype(::TransientImage{T}) where {T} = T
describe(t::TransientImage{T}) where {T} = "Transient.Image($T, ($(t.width), $(t.height)))"
nbytes(t::TransientImage) = t.req.size
# `::Device`, since `::Any` would be ambiguous with `alignment(::Compile, t)`.
alignment(::Device, t::TransientImage) = t.req.alignment
arena(::TransientImage) = Images()
resourcekind(::TransientImage) = ImageKind()

target_view(t::TransientImage) = t.view
target_image(t::TransientImage) = t.image
isdepth(::TransientImage{T}) where {T} = isdepth(T)
target_extent(t::TransientImage) = (Int(t.width), Int(t.height))
target_format(t::TransientImage) = t.format

"""
    Transient.Image(graph, T, (width, height); srgb = false, usage = imageusage(dev, T))
    Transient.Image(graph, T, target)

A render target placed by the compiler, so targets with disjoint lifetimes share
memory. The pixel type `T` is the whole format (`RGBA{Float16}` is
`R16G16B16A16_SFLOAT` on Vulkan). Given a `target` instead of a size, the image
follows that target's extent on resize, e.g. a window's depth buffer
`Transient.Image(g, Float32, screen)`.
"""
function Transient.Image(g::Graph, ::Type{T}, size::Tuple{Integer,Integer};
                         srgb::Bool = false, source = nothing,
                         usage = imageusage(g.dev, T)) where {T}
    width, height = size
    t = makeimage(g.dev, T, Int(width), Int(height), srgb, usage, source)
    push!(g.transients, t)
    t
end

Transient.Image(g::Graph, ::Type{T}, source; kw...) where {T} =
    Transient.Image(g, T, target_extent(source); source, kw...)

"""
    refit!(device, t::TransientImage) -> Bool

Resize a tracking image to its source; `true` if it changed. Images have a fixed
extent, so it is recreated, which invalidates the placement and makes the caller
recompile.
"""
function refit!(dev::Device, t::TransientImage)
    t.source === nothing && return false
    w, h = target_extent(t.source)
    (w, h) == (t.width, t.height) && return false
    t.width, t.height = w, h
    remakeimage!(dev, t)
    t.view = nothing
    t.memory = nothing
    return true
end

"""
    touch!(graph, t)

Record that the newest pass touches `t`. The interval is in declaration order and
provisional; Liveness recomputes it from the scheduled order.
"""
function touch!(g::Graph, t::TransientResource)
    i = length(g.passes)
    t.first = min(t.first, i)
    t.last = max(t.last, i)
    g.transient_by_id[resourceid(g, t)] = t
    t
end

touch!(g::Graph, x) = (resourceid(g, x); x)

"""
    resourcekind(r) -> BufferKind() or ImageKind()

Selects the barrier state machine; only images have layouts.
"""
resourcekind(::Any) = BufferKind()

resourcekind(::WindowSurface) = ImageKind()

"""
    render!(f, graph, name, attachments...; viewport = nothing) -> Pass

A render pass. Each attachment is `target => loadop` (`Clear(value)`, `Keep`,
`Discard`) or a bare target, which keeps its contents. A target with `isdepth`
(a `Float32` image) is the depth attachment, at most one; the others are colour
attachments in order, and a fragment shader returning a tuple writes element `i`
to attachment `i`. A depth-only pass (a shadow map) takes its render area from
the depth target and its fragment shader returns `nothing`. Attachment usages
are recorded before `f` runs, so they come first in the pass.

    render!(g, "scene", screen => Clear(bg), z => Clear(1f0)) do p
        draw!(p, shader, args, n)
    end
"""
function render!(f, g::Graph, name::AbstractString, attachments...; viewport = nothing)
    isempty(attachments) && throw(ArgumentError("a render pass needs a target"))
    p = Pass(name, :render)
    p.viewport = asviewport(viewport)
    push!(g.passes, p)
    for a in attachments
        tgt = a isa Pair ? first(a) : a
        load = a isa Pair ? last(a) : Keep
        load isa LoadOp ||
            throw(ArgumentError("a render target takes Clear(value), Keep or Discard, got $load"))
        touch!(g, tgt)
        if isdepth(tgt)
            p.depth === nothing ||
                throw(ArgumentError("a pass takes one depth target, and this one was given two"))
            p.depth, p.depth_load = tgt, load
            # Read and written depth; `NoAccess` stencil selects the combined
            # depth layout over the separate-aspect ones.
            push!(p.usages, resourceid(g, tgt) => Depth{ReadWrite,NoAccess,discards(load)})
        else
            push!(p.targets, tgt)
            push!(p.loads, load)
            push!(p.usages, resourceid(g, tgt) =>
                  (discards(load) ? ColorAttachment{true} : ColorAttachment{false}))
        end
    end
    f(PassHandle(g, p))
    p
end

stride(a::Attr) = a.stride

count(a::Attr) = count(a.resource)

"""
    Attribute(pass, r) -> Attr

A vertex attribute binding of `r`, declared as a `Vertices` usage of the pass.
This is the only way a resource becomes a vertex binding.
"""
function Attribute(p::PassHandle, r)
    push!(p.pass.usages, resourceid(p.graph, r) => Vertices)
    Attr{eltype(r),typeof(r)}(r, Int32(stride(r)))
end

"""
    drawover(pass, n)

A draw's count: a number, a resource (its length), or a `DrawIndirectCommand`
buffer, which is declared as an `Indirect` usage and read by the device at draw
time.
"""
drawover(p::PassHandle, n) = n

drawover(p::PassHandle, n::Buffer{DrawIndirectCommand}) = indirectcount!(p, n)

drawover(p::PassHandle, n::TransientBuffer{DrawIndirectCommand}) = indirectcount!(p, n)

function indirectcount!(p::PassHandle, n)
    push!(p.pass.usages, resourceid(p.graph, n) => Indirect)
    touch!(p.graph, n)
    Commands(n)
end

"""
    draw!(pass, shader, binding; frag_args, viewport, bindings)

Declare a draw whose count, indices, instances and arguments come from a
[`DrawBinding`](@ref) the host may rewrite between runs. The plan compiles
against its contents at declaration (types fix the pipeline, the texture table
fixes the descriptor layout); each run reads what the cell holds then.
"""
draw!(p::PassHandle, shader, b::DrawBinding; frag_args = (), viewport = nothing,
      bindings = b.bindings) =
    draw!(p, shader, b, b.count; frag_args, viewport, bindings,
          indices = b.indices, instances = b.instances)

function draw!(p::PassHandle, shader, args, n; frag_args = (),
               viewport = nothing, bindings = nothing, indices = nothing,
               instances::Integer = 1)
    # A rebindable cell's contents are what the shader receives and what the
    # draw touches; rebinding keeps the same types and resources.
    va = drawargs(args)
    refuserefs(va)
    refuserefs(frag_args)
    # One push constant range per pipeline: the stages take the same list (Makie
    # objects do), or only one takes any. Checked before `declare!`, whose walk
    # would fail first with a less useful message. `===`, since a list holding a
    # NaN is not `==` to itself.
    isempty(va) || isempty(frag_args) || va === frag_args || throw(ArgumentError(
        "draw!: two DIFFERENT argument lists were given to the two stages, and a " *
        "pipeline has one push constant range. Put them on the stage that reads " *
        "them and pass what the other needs as a varying, or give both stages the " *
        "same list."))
    dev = graphof(p).dev
    isempty(va) || declare!(p, va, vertextouches(dev, shader, va))
    isempty(frag_args) || declare!(p, frag_args, fragmenttouches(dev, shader, frag_args))
    push!(p.pass.draws, DrawCall(shader, args, drawover(p, n), frag_args,
                                 asviewport(viewport), bindings, indices, Int(instances)))
end

storage(v::BufferRange) = storage(v.parent)

# An attribute is a binding, not a resource: its storage is its resource's.
storage(a::Attr) = storage(a.resource)

resourcekind(::BufferRange) = BufferKind()


"""
    slice(graph, x, range) -> BufferRange

Elements `range` of `x` as a resource of its own, interned per
`(resource, range)` in the graph. Kernels still receive the whole buffer; the
slice only scopes barriers, so passes over disjoint slices are not ordered.
For native backend arrays, tracking remains at whole-allocation granularity.
"""
function slice(g::Graph, x, range::UnitRange{Int})
    # Not `length(storage(x))`: a transient has no storage until placement.
    n = length(x)
    (first(range) >= 1 && last(range) <= n) || throw(ArgumentError(
        "slice(): range $range is outside the buffer's 1:$n."))
    # Native arrays may reinterpret one allocation with different element
    # widths/offsets. Track the whole owner until native byte-range views are
    # represented explicitly; element ranges from different dtypes cannot mix.
    isdevicearray(x) && return BufferRange(x,range)
    pid = resourceid(g, x)
    get!(g.views, (pid, range)) do
        BufferRange(x, range)
    end
end

newpass(::Graph, name::AbstractString, kind) = Pass(name, kind)

handle(g::Graph, p::Pass) = PassHandle(g, p)

dispatches(p::Pass) = p.dispatches

passes(g::Graph) = g.passes

"""The graph's first pass, created on demand, that every host store lands in, so
one pair of barriers covers them all."""
function updates_pass!(g::Graph)
    for p in g.passes
        p.kind === :update && return p
    end
    p = Pass("updates", :update)
    pushfirst!(g.passes, p)
    p
end

# What `hostwritten!` collects. A transient only if declared `hostwritten = true`:
# its `CopyDst` usage in the update pass starts its live interval there, so the
# placer cannot alias those bytes with anything live at that point.
hostwritable(::Buffer) = true
hostwritable(::GPURef) = true
hostwritable(t::TransientBuffer) = t.hostwritten
hostwritable(::Any) = false

"""
    hostwritten!(graph) -> Tuple

Register every host-writable resource a pass declares (through `rootresource`)
once as a `CopyDst` of the update pass, and return them as a concretely typed
tuple for `Plan.hostwritten`. Idempotent, so a recompile keeps the registrations.
"""
function hostwritten!(g::Graph)
    found = Any[]
    for p in g.passes
        p.kind === :update && continue
        for (id, _) in p.usages
            r = rootresource(byid(g.ids, id))
            hostwritable(r) || continue
            any(x -> x === r, found) || push!(found, r)
        end
    end
    isempty(found) && return ()
    up = updates_pass!(g)
    for r in found
        id = resourceid(g, r)
        any(u -> u.first == id, up.usages) || push!(up.usages, id => CopyDst)
    end
    return Tuple(found)
end

"""
    copy!(g, name, dst, src; region = nothing) -> Pass

A pass copying image `src` (anything answering `target_image`) into buffer
`dst`, e.g. a framebuffer readback. `region = (x, y, width, height)` copies that
zero-based rectangle, tightly packed; the default is the whole image. Buffer
sources are refused: move buffer bytes with a kernel, so the access walk can
order it.
"""
function copy!(g::Graph, name::AbstractString, dst, src; region = nothing)
    hasmethod(target_image, Tuple{typeof(src)}) || throw(ArgumentError(
        "copy!: `$(name)`'s source is a $(typeof(src)), which is not an image " *
        "attachment — only those answer `target_image`, and this pass records " *
        "`vkCmdCopyImageToBuffer`. To move bytes between BUFFERS, declare the " *
        "kernel that moves them: a dispatch is a pass whose accesses the walk " *
        "reads off the body, and a transfer command is not."))
    p = Pass(name, :copy)
    if region !== nothing
        length(region) == 4 || throw(ArgumentError("copy region must be (x, y, width, height)"))
        x, y, w, h = Int.(region)
        sw, sh = size(src)
        0 <= x && 0 <= y && w > 0 && h > 0 && x + w <= sw && y + h <= sh ||
            throw(ArgumentError("copy region $region lies outside image $(size(src))"))
        p.viewport = (x, y, w, h)
    end
    push!(p.targets, src)
    p.dst = dst
    push!(g.passes, p)
    touch!(g, src); touch!(g, dst)
    push!(p.usages, resourceid(g, src) => CopySrc)
    push!(p.usages, resourceid(g, dst) => CopyDst)
    p
end

argalign(n::Integer) = (Int(n) + 255) & ~255

"""
Stride of one indirect dispatch command (12 bytes used) in the plan's argument
memory, 256 so every backend's indirect-buffer alignment holds.
"""
const INDIRECT_STRIDE = 256

"""
    indirectof(argmemory, k) -> view or nothing

The workgroup-count slot of indirect index `k`, or `nothing` for `k == 0` (a
host-sized dispatch). The views are prebuilt by `ArgMemory`, so this does not
allocate.
"""
indirectof(am::ArgMemory, k::Int) = k == 0 ? nothing : am.indirect[k]

# Whether a compiled dispatch reads its size on the device; `Launch` and `Call`
# answer in `kalaunch.jl`.
devicesized(d::CompiledDispatch) = countresource(d.ndrange) !== nothing

PassPlan(pass, draws, dispatches, images, pre, barrier) =
    PassPlan(pass, draws, dispatches, images, pre, barrier,
             any(devicesized, dispatches))

"""Append `x`, keeping only the last `NSAMPLES`."""
function sample!(ring::Vector{Float64}, x::Real)
    push!(ring, x)
    length(ring) > NSAMPLES && popfirst!(ring)
    ring
end

# `rawargs` resolves Mantle resources to their storage; `devargs` adapts that for
# the device. Look for a TLAS in the raw arguments: adapting strips it, since a
# ray query binds it as a descriptor.
rawargs(args::Tuple) = argmap(storage, args)

devargs(ad, raw::Tuple) = argmap(a -> Adapt.adapt(ad, a), raw)

"""The resource a usage names, following slices and vertex bindings to their parent."""
rootresource(x) = x

rootresource(v::BufferRange) = rootresource(v.parent)

rootresource(a::Attr) = rootresource(a.resource)

Compile(g::Graph; alias = true, policy = Overlap(), ordered_entry = false) =
    Compile(g, alias, policy, ordered_entry, Analysis(),
            IdDict{Pass,Vector{Transition}}(), PassPlan[], Set{Any}())

# Core selects synchronization from native facts and the actual plan route,
# independently of scheduling policy. Walked plans retain first-use barriers.
Compile(pl::Plan) = Compile(pl.graph; pl.alias, pl.policy,
    ordered_entry = buffer_head_ordered(pl.graph.dev) &&
                    recordsplans(pl.graph.dev) && recordable(pl.graph.dev, pl))
ordered_entry(::Compilation) = false
ordered_entry(c::Compile) = c.ordered_entry

# Compilation hooks beyond the generic `Compilation` field accessors: usages,
# slice-aware overlap, and the memory pool.
usages(p::Pass) = p.usages

overlapping(c::Compile, a::Int, b::Int) = overlapping(c.graph, a, b)
overlapgroup(c::Compile, id::Int) = overlapgroup(c.graph, id)

pool(c::Compile) = c.graph.dev.pool

# `nbytes`, `arena` and `describe` have no `TransientResource` fallback, so a new
# transient kind without them is a `MethodError`.

"""
    overlapping(graph, a, b) -> Bool

Whether ids `a` and `b` can name the same bytes: equal ids, a slice and its
parent, or two intersecting slices of one parent.
"""
function overlapping(g::Graph, a::Int, b::Int)
    a == b && return true
    ra, rb = get(g.ids.by_id, a, nothing), get(g.ids.by_id, b, nothing)
    (ra isa BufferRange || rb isa BufferRange) || return false
    # `get`, not `resourceid`: a query must not intern ids while the compiler
    # indexes by id. Parents are registered before their slices.
    pa = ra isa BufferRange ? get(g.ids, ra.parent, 0) : a
    pb = rb isa BufferRange ? get(g.ids, rb.parent, 0) : b
    (pa == 0 || pb == 0) && return false
    pa == pb || return false
    # A whole-resource usage covers every slice of it.
    (ra isa BufferRange && rb isa BufferRange) || return true
    !isempty(intersect(ra.range, rb.range))
end

"""The id of a slice's parent, or `id` itself for anything else."""
function overlapgroup(g::Graph, id::Int)
    r = get(g.ids.by_id, id, nothing)
    r isa BufferRange || return id
    p = get(g.ids, r.parent, 0)
    # An unregistered parent overlaps nothing, so the slice is its own group.
    return p == 0 ? id : p
end

"""
    materialize!(device, t::TransientBuffer, mem, offset)

Give a placed transient its storage. A `BufferBlock` is kept as is, because
Vulkan barriers need the buffer identity, and `storage` builds the array on
demand; any other memory is wrapped once via [`deviceslice`](@ref).
"""
function materialize!(::Device, t::TransientBuffer, blk::BufferBlock, offset::Int)
    t.block, t.offset = blk, offset
    return t
end

function materialize!(dev::Device, t::TransientBuffer{T}, mem, offset::Int) where {T}
    t.block = deviceslice(dev, T, size(t), mem, offset)
    t.offset = offset
    return t
end

"""
    usage_of(pass, id, graph) -> usage type or nothing

What `pass` does to resource `id`, counting overlapping slices, or `nothing`.
The aliasing handover skips its barrier on `nothing`, so a slice must count.
"""
function usage_of(p::Pass, id::Int, g::Graph)
    for (rid, U) in p.usages
        overlapping(g, rid, id) && return U
    end
    nothing
end

"""
    sliceindex(graph) -> Dict{Int,Vector{Int}}

Map each sliced resource's id to its slices' ids, built once per compile because
scanning with `overlapping` per handover is quadratic on large model graphs.
"""
function sliceindex(g::Graph)
    idx = Dict{Int,Vector{Int}}()
    for ((pid, _), v) in g.views
        push!(get!(() -> Int[], idx, pid), resourceid(g, v))
    end
    idx
end

"""
    lastuses(states, slices, id) -> Vector{Type}

The distinct current usages of resource `id` and all of its slices, which live
under different ids; an aliasing handover must wait for all of them.
"""
function lastuses(states::Dict{Int,ResourceState}, slices::Dict{Int,Vector{Int}},
                  id::Int)
    out = Type[]
    take(k) = begin
        st = get(states, k, nothing)
        st === nothing || st.current === nothing || st.current in out ||
            push!(out, st.current)
    end
    take(id)
    for sid in get(slices, id, ())
        take(sid)
    end
    out
end

dispatchrange(n::Integer) = n

dispatchrange(t::Tuple) = t

dispatchrange(x) = count(x)

"""
The index space a `DeviceRange` kernel is compiled against when it gives no
`max`. The host never reads the device count: `__validindex` passes every thread
up to this ceiling and the kernel's own bounds check against the device count
does the rest.
"""
const INDIRECT_CEILING = 1024 * 1024

dispatchrange(r::DeviceRange) = something(r.max, INDIRECT_CEILING)

"""
    checkextents(plan)

Throw if an attachment of a render pass is smaller than the render area (the
first target's extent), e.g. a fixed-size transient after a window resize.
Vulkan leaves that undefined (VUID-VkRenderingInfo-pNext-06079) and RADV renders
garbage silently.
"""
function checkextents(pl::Plan)
    for pp in pl.passes
        p = pp.pass
        p.kind === :render || continue
        want = target_extent(first_target(p))
        for t in (p.targets..., p.depth)
            t === nothing && continue
            e = target_extent(t)
            (e[1] < want[1] || e[2] < want[2]) &&
                error("pass \"$(p.name)\": the render area is $(want[1])x$(want[2]) " *
                      "but an attachment is $(e[1])x$(e[2]). A resize follows the " *
                      "swapchain and not a transient sized when the graph was built — " *
                      "rebuild the graph and the plan at the new size.")
        end
    end
    nothing
end

"""
    Plan(graph; alias = true, profile = false, policy = Overlap(),
         budget = submissionbudget(device))

Compile `graph` into a plan and register it as a tenant of its arenas. Backend
specifics come through [`makeprofiler`](@ref) and [`makeargmemory`](@ref).
`budget` is the estimated GPU seconds per submission the recording is cut to.
"""
Plan(g::Graph; alias::Bool = true,
            profile::Bool = false, policy::Policy = Overlap(),
            budget::Real = submissionbudget(g.dev)) =
    # `hostwritten!` before the compile: it adds the update pass and its `CopyDst` usages.
    let pl = Plan(g, hostwritten!(g), alias, policy, Float64(budget), profile)
        bindreadbacks!(pl)
        c = try
            compile!(Compile(pl))
        catch err
            freereadbacks!(pl, err)
            rethrow()
        end
        install!(pl, c)
        # With timestamps and a finite budget, profile even unasked: the plan
        # measures itself once to partition its recording (`partitionranges`).
        pl.profiler = profile || (timestamps(g.dev) && isfinite(budget)) ?
                      makeprofiler(g.dev, c.passes, true) : nothing
        # Teardown is not finalizer-safe (move listeners, command buffers), so the
        # finalizer queues the plan and `reclaim!` frees it on the owning thread at
        # the next submission.
        finalizer(p -> retireplan!(pool(p.graph.dev), p), pl)
    end

"""
    Graph(device) -> Graph

An empty graph for `device`.
"""
Graph(dev::Device) =
    Graph{typeof(dev)}(dev, Pass[], WindowSurface[], TransientResource[],
                       Dict{Int,TransientResource}(), IdTable(),
                       Dict{Tuple{Int,UnitRange{Int}},Any}(), Any[], false)

"""
    Transient.Buffer(graph, T, dims...; hostwritten = false) -> TransientBuffer

A shaped buffer owned by the graph, placed (and possibly aliased with other
transients) by `Plan`; it has no storage before that. Empty shapes throw.
`hostwritten = true` allows host stores between runs (`t[:] = data`) by keeping
its bytes unaliased from the update pass on; `setindex!` refuses other transients.
"""
Transient.Buffer(g::Graph, ::Type{T}, n::Integer; kw...) where {T} =
    Transient.Buffer(g, T, (Int(n),); kw...)
Transient.Buffer(g::Graph, ::Type{T}, dims::Integer...; kw...) where {T} =
    Transient.Buffer(g, T, map(Int, dims); kw...)
function Transient.Buffer(g::Graph, ::Type{T}, dims::Dims{N};
                          hostwritten::Bool = false) where {T,N}
    checkedbytes(T,dims)
    prod(dims) == 0 && throw(ArgumentError(
        "Transient.Buffer: $(dims) has no elements. An empty result needs no " *
        "buffer and no pass -- skip it where the shape is decided."))
    t = TransientBuffer{T,N}(dims, typemax(Int), 0, nothing, 0; hostwritten)
    t.graph = g
    push!(g.transients, t)
    t
end

npipelines(pl::Plan) = length(pl.pipelines)

"""
    elapsed(lo, hi, period) -> Float64 or nothing

Time between two timestamp ticks, or `nothing` if `hi < lo`. A read racing the
per-frame query pool reset can pair words from two frames, and the `UInt64`
difference would wrap.
"""
elapsed(lo::UInt64, hi::UInt64, period) = hi < lo ? nothing : Float64(hi - lo) * period

"""Whether this plan holds a recording (see `record!`)."""
recorded(pl::Plan) = pl.recording !== nothing

"""
    remappable(plan) -> Bool

Whether the plan has no recording with baked addresses. The block trim
(`movable`) is skipped otherwise, since destroyed blocks cannot be patched; arena
growth instead patches a recorded plan in a buffers arena (`notify_move!`) and
invalidates one in an images arena.
"""
remappable(pl::Plan) = pl.recording === nothing

"""
    invalidate!(pl)

Drop the recording and mark the plan stale so `run!` re-records before its next
submit. Used when an images arena moves (commands name image handles) and after
a refit. No-op for an unrecorded plan. Re-recording is deferred because arena
growth runs on another plan's thread under the pool lock, and only the owning
thread records.
"""
function invalidate!(pl::Plan)
    pl.recording === nothing && return pl
    droprecording!(pl)
    pl.stale = true
    return pl
end

"""
    droprecording!(pl)

Release the recording and clear the patch table, whose offsets point into the
argument memory the next recording rewrites.
"""
function droprecording!(pl::Plan)
    pl.recording === nothing && return pl
    # Unlisten first: a move must not queue a patch into a region `release!`
    # has already returned to the pool.
    unlisten_moves!(pool(pl.graph.dev), pl)
    release!(pl.recording)
    pl.recording = nothing
    empty!(pl.patchtab)
    empty!(pl.pending_patches)
    return pl
end

lp_of(d::CompiledDispatch) = d.launch

"""
    runonce!(g) -> nothing

Plan, record and run `g` once, then free the plan, for one-off work such as
packing weights (there is no immediate-launch API). Does not wait: `free!`
retires the plan's regions. Buffers the graph wrote are not freed. Planning and
recording cost about a millisecond, so recurring work should reuse a plan with
`run!`.
"""
function runonce!(g::Graph)
    plan = Plan(g)
    record!(plan)
    run!(plan)
    free!(plan)
    return nothing
end

"""
    free!(plan)

Drop the recording and retire the plan's transient regions and argument memory
to the pool. Safe right after `run!`: regions are reclaimed only once the device
has passed them. Idempotent; a dropped plan is freed by its finalizer through
`reclaim!`.
"""
function free!(pl::Plan)
    # At most once: an explicit `free!` and the finalizer may both run.
    _, won = @atomicreplace pl.freed false => true
    won || return pl
    # Before the regions: the recording holds a command buffer, descriptor sets
    # and references to every resource it names, and listens for moves.
    droprecording!(pl)
    freereadbacks!(pl)
    let am = pl.args
        am === nothing || retire!(pool(pl.graph.dev), pl.graph.dev, am.store)
    end
    pl.args = nothing
    giveup!(pool(pl.graph.dev), pl.graph.dev, pl.slabs, pl.arenas, pl)
end

"""
    remap!(plan, kind, region)

Re-materialise the plan's transients placed in arena `kind` at the same offsets
in the arena's new `region`.
"""
function remap!(pl::Plan, kind, region)
    ts = pl.graph.transients
    for (i, t) in enumerate(ts)
        # The placement key: `arena(t)` unless split across parts (`ArenaPart`).
        pl.parts[i] == kind || continue
        materialize!(pl.graph.dev, t, memoryof(region), offset(region) + pl.offsets[i])
    end
    # `bake` resolved dispatch arguments over the old region. Unless the backend
    # patches them (`patchable`), mark the plan so `refit!` recompiles before the
    # next submit; recompiling here would re-enter placement mid-growth.
    patchable(pl.graph.dev) || (pl.replaced = true)
    return pl
end


"""
    kernelfor(kernel, group, backend)

The KernelAbstractions kernel object for a launch: `k(backend)`, or
`k(backend, group)` when a group is given (methods in `kalaunch.jl`).
"""
function kernelfor end

"""
    storage(t::TransientBuffer)

The array `materialize!` stored in `t.block`, returned as is, or built on demand
when the block is a [`BufferBlock`](@ref) (Vulkan). Backends specialise on the
block type: a method on `TransientBuffer{T}` itself would overwrite this one.
"""
storage(t::TransientBuffer) = storage(t, t.block)
storage(::TransientBuffer, block) = block
