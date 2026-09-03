# Building a graph: the verbs a caller uses to declare passes.
#
# These name `Graph` and `Pass`, which are concrete in `graph/types.jl`, so they
# could not stay in `runtime/api.jl` — that file is included first and declares
# the vocabulary the graph is built out of, not the graph itself.

"""
    compute!(f, graph, name) -> pass

A pass whose work is `dispatch!` calls. `f` receives the pass handle and declares
what it touches.
"""
function compute!(f, g::Graph, name::AbstractString)
    p = newpass(g, name, :compute)
    push!(passes(g), p)
    f(handle(g, p))
    return p
end

"""
    repeat!(f, graph, maxiters, count) -> passes
    repeat!(f, graph, maxiters; while_nonzero = flag) -> passes

Record `maxiters` iterations of a loop body and let the DEVICE decide how many of
them run. `f(i)` is called once per iteration, `i` in `1:maxiters`, and declares
that iteration's passes exactly as it would outside a loop.

The gate is re-evaluated BEFORE EACH ITERATION, from device memory, which is what
makes this a loop rather than a bounded repeat: the body may move the value the
gate reads, so a loop can end on a condition it discovers as it goes. That is the
shape a wavefront bounce loop has — it runs until the ray queue empties, and
nothing knows when that is until it happens.

Two spellings of the gate, one mechanism:

  * `count` — a device integer; iteration `i` runs while `i <= count[]`.
  * `while_nonzero` — a device integer; the iteration runs while it is not zero.
    A queue's own live count is usually already this, so the loop needs no
    bookkeeping of its own.

Nothing about the trip count reaches the host. The loop is recorded once, at its
maximum, and iterations the gate turns off are discarded at execution.

The lowering is a per-iteration predicate and not device-generated commands, and
the difference is not an implementation detail: DGC repeats a dispatch with NO
barriers between the repetitions, which is right for `count` independent things
and cannot express a loop whose iteration k+1 reads what k wrote.

What a discarded iteration still costs is its barriers, its gate dispatch and its
predicate test; what it saves is the body. So `maxiters` is a bound to be chosen,
not a free parameter — a loop recorded at 1000 and running 3 pays 997 iterations
of that overhead.

The body may not allocate transients that outlive their iteration: every
iteration names the same resources, which is what makes one recording legal.
"""
function repeat!(f, g::Graph, maxiters::Integer, count = nothing;
                 while_nonzero = nothing)
    maxiters >= 1 || throw(ArgumentError("repeat!: maxiters must be at least 1, got $maxiters"))
    (count === nothing) == (while_nonzero === nothing) && throw(ArgumentError(
        "repeat!: give exactly one gate — a `count` positionally, or " *
        "`while_nonzero = flag`."))
    # `typeof` and not the device itself: a device `show`s its whole pool, and an
    # error message that dumps a few hundred kilobytes of zeroed bytes buries the
    # sentence that says what went wrong.
    supportspredicate(g.dev) || throw(ArgumentError(
        "repeat!: $(nameof(typeof(g.dev))) cannot discard recorded work on a " *
        "device-written predicate, so a device-decided trip count is not " *
        "expressible here. Loop on the host, or record a fixed number of " *
        "iterations."))
    n = Int(maxiters)
    src = count === nothing ? while_nonzero : count
    # ONE flag, rewritten before each iteration, rather than an array expanded
    # once up front. The array version was the first design and it cannot express
    # a loop at all: expanding `count` into N flags before the body runs fixes
    # the trip count at a moment when a bounce loop does not yet know it. Writing
    # the flag per iteration costs one small dispatch each and is what lets the
    # body decide whether there is a next one.
    #
    # A single slot is enough BECAUSE the gate is per iteration: the flag is
    # written, read, and written again, and the graph derives both hazards from
    # the usages declared below.
    pred = Buffer(g.dev, [Predicate(0)])
    out = Pass[]
    for i in 1:n
        compute!(g, "repeat!/gate-$i") do p
            use(p, src; read = true)
            use(p, pred; write = true)
            if count === nothing
                dispatch!(p, gate_nonzero!, (pred, src), 1)
            else
                dispatch!(p, gate_count!, (pred, src, Int32(i)), 1)
            end
        end
        first_new = length(passes(g)) + 1
        f(i)
        for k in first_new:length(passes(g))
            pp = passes(g)[k]
            pp.predicate === nothing || throw(ArgumentError(
                "repeat!: pass \"$(pp.name)\" already has a predicate — nested " *
                "`repeat!` is not supported, because a pass has one predicate and " *
                "nesting needs their conjunction."))
            pp.predicate = (pred, 0)
            # The predicate READ is a hazard like any other, and declaring it is
            # what makes the graph put a barrier between the gate that writes the
            # flag and the passes gated on it — and between those passes and the
            # NEXT gate, which overwrites it. Left undeclared both are races the
            # scheduler cannot see.
            push!(pp.usages, resourceid(g, pred) => Predicated)
            touch!(g, pred)
            push!(out, pp)
        end
    end
    return out
end

"""
One iteration's go/no-go flag, as the device reads it.

A distinct type rather than a plain `UInt32` because the ELEMENT TYPE is what
tells a backend how the buffer will be used — the same mechanism
`DrawIndirectCommand` uses to earn `INDIRECT_BUFFER` usage. A predicate buffer
needs a usage flag that no ordinary `UInt32` buffer should carry.

Nonzero runs the iteration, zero discards it. That is the Vulkan convention
(`vkCmdBeginConditionalRenderingEXT` discards on zero) taken as the portable one,
so a kernel writing predicates reads the same on every backend.
"""
struct Predicate
    go::UInt32
end

"""The gate for `repeat!(…, count)`: this iteration runs while `i <= count[]`."""
@kernel function gate_count!(pred, count, i::Int32)
    @inbounds pred[1] = Predicate(i <= Int32(count[1]) ? UInt32(1) : UInt32(0))
end

"""The gate for `repeat!(…; while_nonzero)`: this iteration runs while it is set.

Re-read every iteration, so a body that empties the thing it is draining stops
the loop, and one that refills it carries on."""
@kernel function gate_nonzero!(pred, flag)
    @inbounds pred[1] = Predicate(flag[1] != 0 ? UInt32(1) : UInt32(0))
end

"""
    supportspredicate(device) -> Bool

Whether this device can discard recorded work on a value it reads from a buffer
at execution time — what [`repeat!`](@ref) needs.

`false` in core, so a backend that has no such mechanism refuses `repeat!` at
build rather than silently running every recorded iteration, which would be the
one wrong answer that still produces a picture.
"""
supportspredicate(::Any) = false

"""
    resourceid(graph, r) -> Int

The graph's id for a resource, interning it on first sight.

Here rather than beside the other `resourceid` methods in `runtime/dispatch.jl`
because it names the concrete `Graph`, and that file is included before the
graph exists.
"""
resourceid(g::Graph, r) = resourceid(g.ids, r)


# ── Building and running a graph ──────────────────────────────────────────────
#
# 114 definitions, moved out of `src/vulkan/graph.jl` because not one of them
# names a driver type. They are the graph itself: constructing passes, interning
# resources, deciding what a pass touches, laying out argument memory, walking
# the plan. A backend records and submits what they produce — `graph/backend.jl`
# is the ten functions that takes.
#
# What stayed behind is the other half of that file: barrier emission, command
# recording, and the driver objects those need.

# `Device()` with no argument picks a backend, which core cannot do — it is in
# the Vulkan extension, and a Metal one will add its own.

"""
Usage bits an element type asks for beyond the ordinary ones, which is one type
and one bit: a buffer of draw commands is read by the command processor, and a
buffer without `INDIRECT_BUFFER_BIT` is a validation error at the draw rather
than where it was allocated.

This is `bufferusage`'s answer for this backend — "what memory can host a
`T`" is Vulkan vocabulary, so the backend owns it.
"""
extrausage(::Type) = UInt32(0)

Surface(g, win) = (s = WindowSurface(win); push!(g.surfaces, s); s)

# Forwarded to the WINDOW, which is the backend's.
#
# These four read `s.win.views[s.win.current_image_idx + 1]`, `s.win.extent.width`
# and `s.win.format` directly — a Vulkan swapchain's anatomy, spelled out in the
# shared graph. Nothing else in core knew what a window is made of, and a backend
# whose window is a `CAMetalLayer` handing out one drawable at a time has no
# `views` array and no `current_image_idx` to index it with.
#
# The QUESTIONS are the same four every attachment answers, so a window answers
# them the same way a `TransientImage` does and the render pass cannot tell them
# apart.
target_view(s::WindowSurface)   = target_view(s.win)

target_image(s::WindowSurface)  = target_image(s.win)

target_extent(s::WindowSurface) = target_extent(s.win)

target_format(s::WindowSurface) = target_format(s.win)

# The pixel type, which is what `attachment_format` reads to pick the formats a
# pipeline is compiled for. A window is an attachment like any other and has to
# answer the same question — and core cannot name `BGRA{N0f8}`, so the window
# does.
Base.eltype(s::WindowSurface) = eltype(s.win)

initial_usage(::WindowSurface) = Present

initial_state(::Any) = nothing

initial_state(s::WindowSurface) = initial_usage(s)

DrawCall(shader, args, count) = DrawCall(shader, args, count, ())

# `Pass` is Mantle's now — see `src/graph/types.jl`.

Pass(name, kind) = Pass(String(name), kind, Any[], LoadOp[], nothing, nothing, nothing,
                        DrawCall[], Pair{Int,Type}[], Any[], nothing)

first_target(p::Pass) = isempty(p.targets) ? p.depth : first(p.targets)

# The load op, lowered. `Discard` is the one that needs saying: it is the only
# way to reach DONT_CARE, and a pass that covers every pixel should not pay to
# load what it is about to overwrite.

clearvalue(::LoadOp) = nothing

clearvalue(c::Clear) = NTuple{4,Float32}(c.value)

"""What a depth attachment clears to. One number, not four, and `nothing` loads."""
depthclear(::LoadOp) = nothing

depthclear(c::Clear) = Float32(c.value)

"""What to call a transient in an error, since it has no name of its own."""
describe(t::TransientBuffer{T}) where {T} = "Transient.Buffer($T, $(t.n))"

count(t::TransientBuffer) = t.n

stride(::TransientBuffer) = 1

nbytes(t::TransientBuffer{T}) where {T} = t.n * sizeof(T)

# Asked of the device, not of the transient: both backends place the same
# `TransientBuffer`, and what they need it aligned to differs. 256 is Vulkan's
# `minStorageBufferOffsetAlignment` worst case; the host wants a cache line.
alignment(c::Compile, t) = alignment(c.graph.dev, t)

arena(::TransientBuffer) = Buffers()

Recycler() = Recycler(Dict{Int,Vector{Any}}(), Tuple{Int,Any,Any}[])

"""
Move everything the device has finished with onto the free lists.

Gated on completion tokens rather than on a frame count, because how many runs
are in flight is not this code's business and changing it must not turn
recycling into a use-after-free.

Core's, and it took the move to `passed` to make it so: this asked
`query_timeline(bq)` and compared raw timeline values, which is why it lived in
the Vulkan backend and why `Recycler` held a `UInt64`.
"""
function recycle!(r::Recycler, dev)
    keep = 0
    for (bytes, store, tok) in r.retiring
        if passed(dev, tok)
            push!(get!(() -> Any[], r.free, bytes), store)
        else
            keep += 1
            r.retiring[keep] = (bytes, store, tok)
        end
    end
    resize!(r.retiring, keep)
    r
end

"""Hand a buffer back, reusable once the device has passed `token`."""
retire!(r::Recycler, store, nbytes::Integer, token) =
    push!(r.retiring, (Int(nbytes), store, token))

"""
    imageusage(device, T) -> backend usage mask

What an image of this element type is for, in the backend's own vocabulary.

Asked of the DEVICE, because the answer is a driver value: Vulkan wants
`VkImageUsageFlags`, Metal wants `MTLTextureUsage`, and there is no portable
encoding of "colour attachment" the two could share. What IS portable is the
question — the element type decides, so a target's usage and its aspect cannot
disagree. `Float32` is a depth attachment on both, because it lowers to a
single-component 32-bit depth format on both and nothing else does.

It has to be a hook rather than a default: a colour default naming a Vulkan
constant lived in this file and threw `UndefVarError` the moment a second
backend asked, since the constant only exists when the Vulkan extension loads.
"""
function imageusage end

refit!(::TransientResource) = false

"""
    refit!(plan) -> Bool

Give every tracking transient the size its source now has, and recompile if any
moved.

`true` means the plan is a different plan: different offsets, possibly
different arenas, and a fresh argument memory. A caller that recorded anything
against the old one has to record it again.

Not `any`, which short-circuits — the first transient that moved would be the
only one refitted, and the rest would keep a size their source no longer has.

The caller is `run!`, once per frame, before anything is acquired or recorded:
failing here leaves nothing behind, while the same check inside recording left
a half-recorded frame and an acquired image.
"""
function refit!(pl::Plan)
    moved = false
    for t in pl.graph.transients
        moved |= refit!(t)
    end
    moved || return false
    c = compile!(Compile(pl.graph; pl.alias, pl.coalesce, pl.policy))
    pl.transitions, pl.passes, pl.pipelines = c.transitions, c.passes, c.pipelines
    let a = analysis(c)
        pl.slabs, pl.arenas, pl.offsets = a.regions, a.arenas, a.offsets
        pl.peak, pl.naive = a.peak, a.naive
        # Re-registered: the recompile may have been placed into a different set
        # of arenas, and `tenant!` is idempotent for the ones it was already in.
        for ar in a.arenas
            tenant!(pool(pl.graph.dev), ar, pl)
        end
    end
    # A recompile can change the draws and therefore the layout, so the argument
    # memory is laid out again with them. The old slots may still be read by
    # frames in flight, which is why the outgoing region is RETIRED rather than
    # released: `reclaim!` hands it back one submission boundary later, by which
    # point the frames that named it have run.
    let am = pl.args
        am === nothing || retire!(pool(pl.graph.dev), am.store)
    end
    pl.args = makeargmemory(pl.graph.dev, c.passes)
    return true
end

# ── Transient render targets ─────────────────────────────────────────────────

Base.size(t::TransientImage) = (t.width, t.height)
Base.eltype(::TransientImage{T}) where {T} = T
describe(t::TransientImage{T}) where {T} = "Transient.Image($T, ($(t.width), $(t.height)))"
nbytes(t::TransientImage) = t.req.size
# `::Device`, not `::Any`: `alignment(c::Compile, t)` above would be ambiguous
# with an unconstrained first argument, and a `Compile` is not a device.
alignment(::Device, t::TransientImage) = t.req.alignment
arena(::TransientImage) = Images()
resourcekind(::TransientImage) = ImageKind()

target_view(t::TransientImage) = t.view
target_image(t::TransientImage) = t.image
isdepth(::TransientImage{T}) where {T} = isdepth(T)
target_extent(t::TransientImage) = (Int(t.width), Int(t.height))
target_format(t::TransientImage) = t.format
# Nothing has been written into it and after aliasing it is back in that state.
initial_usage(::TransientImage) = Undefined
initial_state(t::TransientImage) = initial_usage(t)

"""
    Transient.Image(graph, T, (width, height); srgb = false, usage = imageusage(dev, T))
    Transient.Image(graph, T, target)

A render target the compiler places, so two targets whose lifetimes do not
overlap share bytes.

`T` is the pixel type and is the whole format: `RGBA{Float16}` is
`R16G16B16A16_SFLOAT` on Vulkan and `MTLPixelFormatRGBA16Float` on Metal.
Readback of one gives a `Matrix{T}` and a kernel writing one writes `T`, which
an enum would not have given.

Given a *target* rather than a size, it follows that target: a depth buffer for a
window is `Transient.Image(g, Float32, screen)`, and it is resized with the
window. A fixed size is the right answer only when it genuinely is fixed, because
an attachment that stops covering the render area is undefined rendering.
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
Give a tracking transient the size its source now has, and say whether it moved.

The image is recreated rather than resized, because an image has its extent from
creation on every backend this runs on. That invalidates the placement — a
different size needs different offsets — which is why the caller recompiles
rather than patching the old plan.
"""
function refit!(t::TransientImage)
    t.source === nothing && return false
    w, h = target_extent(t.source)
    (w, h) == (t.width, t.height) && return false
    t.width, t.height = w, h
    remakeimage!(t)
    t.view = nothing
    t.memory = nothing
    return true
end

"""
Record that this pass touches `t`.

The interval recorded here is in declaration order and is provisional: Liveness
recomputes it from the scheduled order, because reordering passes is exactly what
changes a transient's lifetime.
"""
function touch!(g::Graph, t::TransientResource)
    i = length(g.passes)
    t.first = min(t.first, i)
    t.last = max(t.last, i)
    g.transient_by_id[resourceid(g, t)] = t
    t
end

touch!(g::Graph, x) = (resourceid(g, x); x)

# `resourceid(g::Graph, r)` is NOT defined here. `resourceid(g::Graph, r)`
# (runtime/dispatch.jl) is the same line, `Graph <: Graph`, and Mantle
# exports it — so this was a SECOND function of the same name in this module,
# agreeing with core's by coincidence rather than by construction.

"""
What kind of resource an id names, so barrier tracking starts in the right state
machine. Images have layouts and buffers do not, and a buffer tracked as an image
would emit layout transitions for a resource that has none.
"""
resourcekind(::Any) = BufferKind()

resourcekind(::WindowSurface) = ImageKind()

"""
A render pass. `target => clear` clears; a bare target loads what is there.

The target is a usage like any other, recorded before the body runs so it is
first in the sequence. Deriving it from `initial_usage` instead would be a second
statement of the same thing, and would be wrong the moment two passes render to
one target: the second would transition from the target's declared initial state
rather than from the colour attachment the first left it in.

A depth target is another attachment, spelled the same way, and which slot an
attachment fills comes from the target rather than from its position or a keyword:

    z = Transient.Image(g, Float32, (w, h))
    render!(g, "scene", screen => Clear(bg), z => Clear(1f0)) do p

`z` is a depth attachment because a `Float32` image is `D32_SFLOAT` and nothing
else in Vulkan is a single-component 32-bit float attachment. Its barrier and its
layout are then derived from the `Depth` usage like any other.

Several colour attachments are the same list again, in order, and the fragment
shader writes them by returning a tuple — element `i` goes to attachment `i`:

    render!(g, "gbuffer", albedo => Clear(bg), normals => Discard, z => Clear(1f0)) do p
        draw!(p, GBUF, bind(p, mesh, mvp), mesh.positions)   # fragment returns (c, n)
    end

A pass with only a depth target is a pass all the same, and it is what a shadow
map is: nothing is shaded, and the render area comes from the depth attachment
because there is nothing else to take it from. Its fragment shader returns
`nothing`, which is how a shader says it writes no attachment.

    render!(g, "shadow", shadowmap => Clear(1f0)) do p
        draw!(p, DEPTHONLY, args, drawcmd)
    end
"""
function render!(f, g::Graph, name::AbstractString, attachments...)
    isempty(attachments) && throw(ArgumentError("a render pass needs a target"))
    p = Pass(name, :render)
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
            # Tested and written, and no stencil aspect: `NoAccess` is what says
            # the image has none, which is what picks the combined layout over the
            # separate-aspect ones.
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

# `PassHandle` is Mantle's now — see `src/graph/types.jl`.

# `Attr` is Mantle's now — see `src/graph/types.jl`.

stride(a::Attr) = a.stride

count(a::Attr) = count(a.resource)

"""Constructing a usage is how a pass gets a binding. There is no path from a
resource to a draw that skips it."""
function Attribute(p::PassHandle, r)
    push!(p.pass.usages, resourceid(p.graph, r) => Vertices)
    Attr{eltype(r),typeof(r)}(r, Int32(stride(r)))
end

# `Commands` is Mantle's now — see `src/graph/types.jl`.

"""
What a draw takes its vertex count from.

A number is one. A resource is its length, so a draw over a buffer of points
covers exactly the points there are. A buffer of `DrawIndirectCommand` is the
third answer and the only one the host never learns: the count is read by the
command processor from device memory, so a compute pass in the same frame can
decide it. The element type is what says which, because a buffer of draw commands
is not something anything else would be.
"""
drawover(p::PassHandle, n) = n

drawover(p::PassHandle, n::Buffer{DrawIndirectCommand}) = indirectcount!(p, n)

drawover(p::PassHandle, n::TransientBuffer{DrawIndirectCommand}) = indirectcount!(p, n)

function indirectcount!(p::PassHandle, n)
    push!(p.pass.usages, resourceid(p.graph, n) => Indirect)
    touch!(p.graph, n)
    Commands(n)
end

function draw!(p::PassHandle, shader, args, n; frag_args = ())
    # Here rather than at compile: the pipeline has one push constant range, so a
    # draw with arguments on both stages is a mistake in the call, and by the time
    # a shader is compiled it surfaces as one stage failing to take an argument it
    # never declared.
    isempty(args) || isempty(frag_args) || throw(ArgumentError(
        "draw!: arguments were given to both stages, and a pipeline has one push " *
        "constant range. Put them on the stage that reads them and pass what the " *
        "other needs as a varying."))
    push!(p.pass.draws, DrawCall(shader, args, drawover(p, n), frag_args))
end

# `BufferRange` is Mantle's now — see `src/graph/types.jl`.

storage(v::BufferRange) = storage(v.parent)

# An attribute is a BINDING of a resource, not a resource — its storage is
# whatever it was made from. Beside `BufferRange` for the same reason: both are
# views, and `rootresource` already forwards through both.
storage(a::Attr) = storage(a.resource)

resourcekind(::BufferRange) = BufferKind()


function slice(g::Graph, x, range::UnitRange{Int})
    # `length`, not `length(storage(x))`: a transient has no storage until the
    # placer gives it some, and a range is declared while the graph is built.
    n = length(x)
    (first(range) >= 1 && last(range) <= n) || throw(ArgumentError(
        "use(): range $range is outside the buffer's 1:$n."))
    pid = resourceid(g, x)
    get!(g.views, (pid, range)) do
        BufferRange(x, range)
    end
end

"""
    use(pass, x; read, write, range = nothing, unordered = false)

The ordinary case: this pass reads or writes this resource. Named usages survive
only where the role can be picked wrongly.

`range` narrows the claim to a slice, in elements. Two passes that name disjoint
slices of one buffer get no barrier between them, and one that does name a slice
gets a barrier scoped to exactly those bytes.

`unordered` says the order of this access against another unordered one does not
change the result — commutative atomics, or writes to disjoint elements. Two
passes that BOTH say it get no barrier between them however much they overlap;
anything else still does, so forgetting it anywhere gives the barrier back rather
than producing a race. A wavefront tracer's per-pixel radiance is the case it
exists for: half a dozen stages do nothing to it but `atomic +=`, and ordering
them against each other serialises passes whose queues are disjoint.
"""
function use(p::PassHandle, x; read::Bool = false, write::Bool = false,
                    range::Union{Nothing,UnitRange{Int}} = nothing,
                    unordered::Bool = false)
    read || write || throw(ArgumentError("use() needs read, write, or both"))
    S = Storage{BufferKind, Access{read, write}}
    U = unordered ? Unordered{S} : S
    r = range === nothing ? x : slice(p.graph, x, range)
    push!(p.pass.usages, resourceid(p.graph, r) => U)
    # The parent is what the kernel gets, and what liveness has to see touched.
    touch!(p.graph, x)
end

# Whether a dispatch was given a workgroup size is a type, not a branch: the
# launch is per pass per frame and this keeps the call site one expression.

# `kernelfor` names a backend's KA backend object, so its methods are the
# backend's — see the Vulkan and host ones. Only the generic is core's.



# The body goes in `dispatches` beside the `Dispatch`es rather than in a field of
# its own: the compile walks that vector and this is one more thing it can find
# there, so `Pass` does not grow a field only one kind ever sets.
# The four hooks core's `custom!`/`compute!` are written against.

newpass(::Graph, name::AbstractString, kind) = Pass(name, kind)

handle(g::Graph, p::Pass) = PassHandle(g, p)

dispatches(p::Pass) = p.dispatches

passes(g::Graph) = g.passes

"""The single pass every `Update` shares, so one pair of barriers covers them all."""
function updates_pass!(g::Graph)
    for p in g.passes
        p.kind === :update && return p
    end
    p = Pass("updates", :update)
    pushfirst!(g.passes, p)
    p
end

function Update(g::Graph, buf; range = nothing)
    p = updates_pass!(g)
    touch!(g, buf)
    return registerupdate!(g.updates, p.usages, resourceid(g, buf), buf, range)
end

# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
function copy!(g::Graph, name::AbstractString, dst, src)
    p = Pass(name, :copy)
    push!(p.targets, src)
    p.dst = dst
    push!(g.passes, p)
    touch!(g, src); touch!(g, dst)
    push!(p.usages, resourceid(g, src) => CopySrc)
    push!(p.usages, resourceid(g, dst) => CopyDst)
    p
end

# ── plan ──────────────────────────────────────────────────────────────────────
# `CompiledDraw` is Mantle's now — see `src/graph/types.jl`.

# `CompiledDispatch` is Mantle's now — see `src/graph/types.jl`.

# `ArgMemory` is Mantle's now — see `src/graph/types.jl`.

argalign(n::Integer) = (Int(n) + 255) & ~255

slotbase(am::ArgMemory) = (am.slot - 1) * am.stride

"""
One `VkDispatchIndirectCommand`'s worth of the plan's memory, 256-byte aligned so
a device address derived from it satisfies every backend's indirect-buffer
alignment. Twelve bytes are used; the alignment is what decides the stride.
"""
const INDIRECT_STRIDE = 256

"""
Where this dispatch's workgroup counts live in the slot that is current, or
`nothing` if the host already knows its ndrange.

The one lookup a recording does per device-sized dispatch, and it is an index
rather than an allocation: `ArgMemory` built the view when it laid the slot out.
"""
indirectof(am::ArgMemory, k::Int) = k == 0 ? nothing : am.indirect[am.slot][k]

# ── lowering a transition into a Vulkan barrier ───────────────────────────────

# `indirect` is asked of a recorded dispatch, which has an `ndrange`. A KA
# backend's baked callable does not — it resolved its count into the closure — so
# the probe is `hasproperty`-guarded rather than assuming the recording shape.
PassPlan(pass, draws, dispatches, images, pre, barrier) =
    PassPlan(pass, draws, dispatches, images, pre, barrier,
             any(d -> hasproperty(d, :ndrange) && d.ndrange isa DeviceRange,
                 dispatches))

# `Profiler` is Mantle's now — see `src/graph/types.jl`.

"""Keep the last `NSAMPLES`, so a plan that runs for an hour does not grow."""
function sample!(ring::Vector{Float64}, x::Real)
    push!(ring, x)
    length(ring) > NSAMPLES && popfirst!(ring)
    ring
end

# Mutable because a window resize re-places every transient, and what comes back
# is a different set of slabs and different barriers. The plan is the user's
# handle, so it is updated rather than replaced.
# `Plan` is Mantle's now — see `src/graph/types.jl`.

# `adaptor` builds a backend's argument adaptor; it is the backend's.

# In two steps, and the order matters for one thing: an acceleration structure.
# `rawargs` is what the caller gave, with `Ref`s read and Mantle's own resources
# resolved to their storage; `devargs` is that after Lava's conversion. A ray
# query needs the `VulkanTLAS` bound as a descriptor, and adapting an
# `AdaptedAccel` deliberately strips it — the device side of a ray query is a
# variable, not a pointer the kernel carries. So the HWTLAS is looked for in the
# RAW arguments, which is where it still exists. Asking the adapted ones finds
# nothing, and a shading kernel then compiles with ray query disabled and fails
# in the emitter rather than at the call site.

rawargs(args::Tuple) = map(argvalue, args)

"""
Whether this argument can hold a different value on the next run.

The mirror of [`argvalue`](@ref), and deliberately the same one level deep:
`rawargs` maps `argvalue` over the argument tuple and `argvalue` unwraps exactly
one thing, a `Base.RefValue`. So a `Ref` moves and nothing else does, and this
says so with the same two methods rather than a second opinion about it.

What it is for: a recorded plan submits commands that read their arguments out of
the plan's own memory, so the only per-run host work it needs is rewriting the
arguments that can differ. Anything else was written at `record!` and is still
there. See `Plan.writes`.
"""
isdynamic(::Base.RefValue) = true

isdynamic(x) = false

"""Whether any argument in a launch's tuple can move. See [`isdynamic`](@ref)."""
dynamicargs(args::Tuple) = any(isdynamic, args)

# Per compiled entry, and by dispatch rather than by a field name that happens to
# be shared: a draw counts vertices and a dispatch and a trace take an ndrange,
# and the three types name that field differently on purpose.
rebinding(d::CompiledDraw) = dynamicargs(d.args) || isdynamic(d.count)

rebinding(d::CompiledDispatch) = dynamicargs(d.args) || isdynamic(d.ndrange)

rebinding(t::CompiledTrace) = dynamicargs(t.args) || isdynamic(t.ndrange)

devargs(ad, raw::Tuple) = map(a -> Adapt.adapt(ad, a), raw)

"""The resource a usage ultimately names: a slice and a vertex binding both
forward their storage to a parent, so neither has an identity of its own to test."""
rootresource(x) = x

rootresource(v::BufferRange) = rootresource(v.parent)

rootresource(a::Attr) = rootresource(a.resource)

"""
Whether an `Update` on this resource can move it.

Only the whole-buffer route renames: `write_update!` renames when the ref has no
range *and* the data is the buffer's whole length, and writes in place otherwise.
So an `Update(g, buf; range = 1:100)` never moves its target and keeps a scoped
barrier; a bare `Update(g, buf)` may, and gives it up.

Through a slice as well as directly: `slice(g, buf, 1:100)` is a different object
from `buf`, so an identity test against the update refs misses it — and a slice of
a renamed buffer is exactly as stale as the buffer, having no storage of its own.

Asked of the graph rather than of the resource because the resource cannot know:
a `LavaBuffer` is the same type either way, and whether it is renameable is a
property of how the graph was declared.
"""
renameable(g::Graph, r) =
    any(u -> u.resource === rootresource(r) && u.range === nothing, g.updates)

Compile(g::Graph; alias = true, coalesce = true, policy = Overlap()) =
    Compile(g, alias, coalesce, policy, Analysis(), Transition[],
            IdDict{Pass,Vector{Transition}}(), PassPlan[], Set{Any}())

# What core's phases ask of a compilation context, and it is now only what
# DIFFERS from every other backend. The seven field accessors — `analysis`,
# `passes`, `policy`, `alias`, `transients`, `transientbyid`, `device` — were
# written verbatim here and in `MantleHostExt`, so they come from
# `Compilation` and neither writes them.
#
# These three stay because they are genuinely this backend's: how a pass records
# its usages, whether two ids can name the same bytes (this backend HAS slices,
# so it is not `a == b`), and which pool the memory comes from.

usages(p::Pass) = p.usages

overlapping(c::Compile, a::Int, b::Int) = overlapping(c.graph, a, b)

pool(c::Compile) = c.graph.dev.pool

# `nbytes`, `arena` and `describe` are answered per concrete transient, above.
#
# There were `::TransientResource` fallbacks here that read `nbytes(t) = nbytes(t)`
# — each one calling itself. Unreachable while every transient is a `TransientBuffer`
# or a `TransientImage`, both of which have a specific method, and an infinite
# recursion the moment a third kind is added. Deleted rather than fixed: a
# fallback that cannot say anything useful should be a MethodError naming the
# type that has not answered.

# ── Dag ───────────────────────────────────────────────────────────────────────

writes_it(U) = writes(U)

"""
Whether two resource ids can name the same bytes.

Equal ids do. So does a slice against its own parent, and two slices of one
parent whose ranges intersect — those are different ids, and a scheduler told
they are unrelated is free to reorder two passes that write the same memory.
"""
function overlapping(g::Graph, a::Int, b::Int)
    a == b && return true
    ra, rb = get(g.ids.by_id, a, nothing), get(g.ids.by_id, b, nothing)
    (ra isa BufferRange || rb isa BufferRange) || return false
    # `get`, not `resourceid`: this answers a question and must not hand out an
    # id doing it. A slice's parent is always registered first — `use` touches it
    # before `slice` is reached — so the fallback is unreachable, but a query that
    # can grow `by_id` while the compiler is indexing by id is not worth leaving
    # to that invariant holding.
    pa = ra isa BufferRange ? get(g.ids.ids, ra.parent, 0) : a
    pb = rb isa BufferRange ? get(g.ids.ids, rb.parent, 0) : b
    (pa == 0 || pb == 0) && return false
    pa == pb || return false
    # A whole-resource usage covers every slice of it.
    (ra isa BufferRange && rb isa BufferRange) || return true
    !isempty(intersect(ra.range, rb.range))
end

"""Give a placed transient its storage."""
function materialize!(t::TransientBuffer, blk::BufferBlock, offset::Int)
    t.block, t.offset = blk, offset
    return t
end

"""
What a pass does to one resource, or `nothing` if it does not name it.

A slice counts as naming its parent. The handover asks this about a *transient*,
and a pass that names only a slice of one would otherwise answer `nothing` — on
which the caller skips the barrier entirely, which is the hazard no per-resource
sequence can see going missing without a word.
"""
function usage_of(p::Pass, id::Int, g::Graph)
    for (rid, U) in p.usages
        overlapping(g, rid, id) && return U
    end
    nothing
end

"""
Every id one resource is tracked under: itself, plus each slice of it.

Built once per compile rather than rediscovered per handover. `lastuses` used to
find these by scanning every tracked state and asking `overlapping`, which is
O(resources) inside a per-pass loop — invisible on a twenty-pass render graph and
1.75 s of a 2 s compile at fourteen hundred, which is the scale a model graph
arrives at. Resources without slices get an empty entry and the O(1) path.
"""
function sliceindex(g::Graph)
    idx = Dict{Int,Vector{Int}}()
    for ((pid, _), v) in g.views
        push!(get!(() -> Int[], idx, pid), resourceid(g, v))
    end
    idx
end

"""
Everything the old tenant was last doing, across however many ids it is tracked
under. Whole and sliced usages of one transient live under different ids, and the
handover has to wait for all of them, not for whichever the transient itself
happens to be keyed by.
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

# ── Barriers ──────────────────────────────────────────────────────────────────

dispatchrange(n::Integer) = n

dispatchrange(t::Tuple) = t

dispatchrange(x) = count(x)

# A `DeviceRange` is never read here: the count lives on the device and reading
# it would be the host readback the whole mechanism exists to avoid. The kernel
# is compiled against a CEILING so `__validindex` lets every thread through, and
# the real bound is the count the GPU reads at dispatch time — the kernel's own
# `i <= n` check does the rest. Same contract as `ka_launch_indirect!`,
# which is what records it.

"""
The thread count a `DeviceRange` kernel is COMPILED against, when it declares no
`max` of its own.

Here and not in the backend, which is where it stayed when the `dispatchrange`
methods around it moved to core: it is a compile-time bound on a kernel's index
space, and every backend needs the same one. Left behind it was an
`UndefVarError` from core the first time anything dispatched over a device-side
count — which is every Hikari frame, and nothing before that.
"""
const INDIRECT_CEILING = 1024 * 1024

dispatchrange(r::DeviceRange) = something(r.max, INDIRECT_CEILING)

"""
    Plan(graph; alias = true, coalesce = true, profile = false, policy = Overlap())

Compile `graph` into a plan.

Generic over the backend. Compiling, reading the analysis out and registering
the plan as a tenant of every arena it was placed into are the same on all of
them; the two things that are not are hooks — [`makeprofiler`](@ref) and
[`makeargmemory`](@ref) — which default to `nothing` for a backend that has
neither.
"""
Plan(g::Graph; coalesce::Bool = true, alias::Bool = true,
            profile::Bool = false, policy::Policy = Overlap()) =
    let c = compile!(Compile(g; alias, coalesce, policy))
        let a = analysis(c)
            pl = Plan(g, c.transitions, c.passes, c.pipelines, a.regions, a.arenas,
                          a.offsets, a.peak, a.naive,
                          makeprofiler(g.dev, c.passes, profile),
                          alias, coalesce, policy,
                          makeargmemory(g.dev, c.passes), nothing, nothing)
            # After construction, because a plan cannot be a tenant before it is a
            # plan — and the arena it was just placed into may grow for the NEXT
            # plan, which is when this registration earns its keep.
            for ar in a.arenas
                tenant!(pool(g.dev), ar, pl)
            end
            pl
        end
    end

"""
    Graph(device) -> Graph

An empty graph for `device`.

Generic: a graph holds passes, surfaces, transients and an id table, and not one
of those is a backend's. Both backends defined this identically before it moved
here.
"""
Graph(dev::Device) =
    Graph{typeof(dev)}(dev, Pass[], WindowSurface[], TransientResource[],
                       Dict{Int,TransientResource}(), IdTable(), Any[], Recycler(),
                       Dict{Tuple{Int,UnitRange{Int}},Any}())

function Transient.Buffer(g::Graph, ::Type{T}, n::Integer) where {T}
    t = TransientBuffer{T}(Int(n), typemax(Int), 0, nothing, 0)
    push!(g.transients, t)
    t
end

# `overlapping(::Graph, a, b)` is above, with the Dag phase. It is already
# generic: a backend that never creates a `BufferRange` falls through to
# `a == b`, which is the floor its docstring describes.

# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
npipelines(pl::Plan) = length(pl.pipelines)

"""
Nanoseconds between a pass's two timestamps, or `nothing` if the pair cannot be
from one frame.

The queries are `UInt64`, so an end that precedes its start does not come out
negative — it wraps to about 1.8e19 ns, which `timings` then reports as a
hundred billion milliseconds. That happens: the pool is reset at the head of
every frame's recording, and a read that races the reset can take the start word
from one frame and the end word from the next, with both availability bits set.
There is no time to report for such a pair, so it is dropped rather than
averaged in.
"""
elapsed(lo::UInt64, hi::UInt64, period) = hi < lo ? nothing : Float64(hi - lo) * period

# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
"""Whether this plan's commands have been written — see `record!`."""
recorded(pl::Plan) = pl.recordings !== nothing

"""A recorded plan's command buffers hold the addresses its region has today, so
the arena it is placed in can no longer grow. See `remappable`."""
remappable(pl::Plan) = pl.recordings === nothing

# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
lp_of(d::CompiledDispatch) = d.launch

# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
# ↓ moved to src/vulkan/graph.jl — it names this backend's command queue.
"""
Give this plan's regions back to the pool — its transients', and the argument
memory its recordings read. The pipelines are ordinary backend objects and the
GC reclaims those; a region is the thing only an explicit call can return,
because nothing here finalizes.

No precondition: the regions are retired, so a plan freed immediately after its
last `run!` — the ordinary case, with its recording still in flight — is fine.
"""
function free!(pl::Plan)
    # The recordings too, and before the regions: each holds a command buffer, a
    # descriptor set per acceleration structure it binds, and a reference to
    # every resource its commands name. Dropping the plan alone would leave all
    # of that to the GC, which does not know it is holding device memory.
    if pl.recordings !== nothing
        for rec in pl.recordings
            release!(rec)
        end
        pl.recordings = nothing
        pl.writes = nothing
    end
    # The argument memory is a region like the transients' are, so it goes back
    # the same way. It used to be a device array from the backend's own allocator
    # and was left to the GC, which is what "the argument memory … the GC reclaims
    # those" above meant; there is one owner now and it is this call.
    let am = pl.args
        am === nothing || retire!(pool(pl.graph.dev), am.store)
    end
    pl.args = nothing
    giveup!(pool(pl.graph.dev), pl.slabs, pl.arenas, pl)
end

"""Re-materialise this plan's transients of `kind` into the arena's new region.
Its own offsets are unaffected by the arena moving; only the base changed."""
function remap!(pl::Plan, kind, region)
    ts = pl.graph.transients
    for (i, t) in enumerate(ts)
        arena(t) == kind || continue
        materialize!(t, memoryof(region), offset(region) + pl.offsets[i])
    end
    return pl
end


"""
    kernelfor(kernel, group, backend)

The KernelAbstractions kernel object for a launch, from whichever backend is
running it.

Declared here, answered by each backend: `k(backend)` and `k(backend, group)`
are the same two lines everywhere, but `backend` is the backend's own KA object.
"""
function kernelfor end

"""
A materialised transient's storage, from whatever `materialize!` put in `block`.

Two backends out of three put the finished array there — a `Vector` view on the
host, a borrowed `MtlArray` on Metal — and hand it straight back. Vulkan cannot:
`block` holds a [`BufferBlock`](@ref), because placement needs the
`VkBuffer` identity and an array view does not carry it, so the `LavaArray` is
built over the block on demand.

Dispatch on the block and not one method per backend, which is what this was:
the backend wrote `storage(t::TransientBuffer{T}) where {T}`, the same signature
as an untyped `TransientBuffer`, so it OVERWROTE this rather than specialising
it — fatal during precompilation, and had it loaded, whichever module was
included last would have decided the answer for every backend at once.
"""
storage(t::TransientBuffer) = storage(t, t.block)
storage(::TransientBuffer, block) = block
