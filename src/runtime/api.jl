# The runtime surface. Generic here; a backend extension implements it.
#
# Lifetime is in the name, not in an argument: `Buffer(dev, ...)` is persistent
# and you update! it, `Transient.Buffer(g, ...)` is the compiler's. Neither is
# ever freed by hand.

abstract type Device end
abstract type Resource end
abstract type Graph end
abstract type Plan end

"""
    DeviceCaps

What a kernel has to know about the GPU it will run on, in terms every GPU has.

Nothing here is Vulkan's. The vocabulary is — "subgroup" is SPIR-V's word, Metal
says simdgroup and CUDA says warp; "workgroup" is SPIR-V's, Metal says threadgroup
— but every field is a fact about the hardware that each of them reports under
some name. Kept in that vocabulary rather than renamed, because a kernel library
already speaks it and a rename buys nothing a comment cannot.

Why it lives here and not in the backend: a kernel that picks its tiling from
these numbers is portable exactly to the extent that these numbers are, and the
whole point of the `tile` field is that Metal's `simdgroup_matrix` is 8x8 where
RDNA 3.5 is 16x16 — a different *number*, not a different kernel.

`coopmat` is a floor, not a promise about every operation: cooperative matrices
that exist everywhere are load, store and multiply-add. Anything narrower —
per-element application, in-tile reduction — is `VK_NV_cooperative_matrix2` and
not portable even across Vulkan, so a kernel that wants it asks separately and
carries the answer. See `FlashCMPlan.rescale` for the shape that takes.

    coopmat           cooperative-matrix multiply-add is usable at all
    tile              its tile extent: 16 on RDNA 3.5, 8 for Metal's simdgroup
    subgroup          lanes per subgroup — 32 on NVIDIA, 32 or 64 on RDNA3
    coopmatsubgroup   …and the width a cooperative-matrix kernel actually gets
    sharedbudget      bytes of workgroup-shared memory
    workgrouplimit    threads per workgroup
    cores             SMs / CUs; 0 when the device will not say
    warps             max resident subgroups per core; 0 = ditto
    wggran            workgroup-scope matrix shapes: (invocations, M, N, K) rows

`wggran` is the one field that is a *table* rather than a number, and it is a
table because the answer depends on the launch: the legal `(M, N, K)` multiples
for a matrix spanning the whole workgroup differ per workgroup size, and coarsen
as it grows. Empty means the device has no workgroup-scope matrices, so it is
also the capability test — a kernel cannot learn "may I?" without learning "at
what shapes?", which is the pair that must not drift apart.
"""
struct DeviceCaps
    coopmat::Bool
    tile::Int
    subgroup::Int
    coopmatsubgroup::Int
    sharedbudget::Int
    workgrouplimit::Int
    cores::Int
    warps::Int
    wggran::Vector{NTuple{4,Int}}
end

# Eight positional arguments still construct one, meaning "no workgroup-scope
# matrices" — every caller that predates `wggran` says exactly that.
DeviceCaps(coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
           workgrouplimit, cores, warps) =
    DeviceCaps(coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
               workgrouplimit, cores, warps, NTuple{4,Int}[])

"""
    DeviceCaps(c; kw...) -> DeviceCaps

`c` with named fields replaced, for asking what a kernel would decide on a device
that is not this one — a wave64 card, or this card with cooperative matrices
switched off — without that device being present. It is what makes a tiling
decision testable on a machine that cannot run it.
"""
DeviceCaps(c::DeviceCaps;
           coopmat = c.coopmat, tile = c.tile, subgroup = c.subgroup,
           coopmatsubgroup = c.coopmatsubgroup, sharedbudget = c.sharedbudget,
           workgrouplimit = c.workgrouplimit, cores = c.cores, warps = c.warps,
           wggran = c.wggran) =
    DeviceCaps(coopmat, tile, subgroup, coopmatsubgroup, sharedbudget,
               workgrouplimit, cores, warps, wggran)

"""
    caps(device) -> DeviceCaps

What this device can do. A backend implements it; nothing above it needs to know
which backend answered.

The backend converts rather than aliases: Lava has a struct of its own with the
same fields, and it stays Lava's. Mantle cannot be a dependency of its own
backend, so the type has to be defined here and filled in there.
"""
function caps end

"""
    Window(width, height; title = "", vsync = false)

Something to render into, and the only reason a frame loop exists.

    win = Window(1000, 750; title = "particles")
    while isopen(win)
        run!(plan)
    end
    close(win)

`isopen` is a predicate and nothing more: events are pumped by `run!`, once per
frame, for every surface the plan draws to. A loop that asks whether the window
is open and then draws therefore behaves the way it reads, and nothing has to
remember to poll.

There is no `flush` in that loop, and no `present` either. `run!` submits the
frame and the next one waits on the fence for its own slot, which is the pacing;
a wait for the GPU to go idle would only stop the CPU from working ahead. A
readback synchronises itself, because it must.
"""
abstract type Window end

"""
    backend(device)

The KernelAbstractions backend behind a device, for kernels launched outside a
graph — data produced by something the graph knows nothing about and then handed
to an `Update`.

    advect!(backend(dev))(positions, velocities, dt; ndrange = n)

Inside a graph, `dispatch!` is the way: the graph then knows the kernel wrote
that buffer, and orders the passes that read it.
"""
function backend end

"""
    screenshot(window) -> Matrix

The pixels currently on the window, as a matrix of its format's element type.
Synchronises internally, and returns the image it acquired to the presentation
engine rather than leaving it outstanding.
"""
function screenshot end

"""
What a render pass does with whatever is already in its target.

Three answers, not two. `Keep` loads, `Clear` overwrites with a value, and
`Discard` says the pass covers every pixel and the old contents are worth
nothing. Only the last is free: it is `LOAD_OP_DONT_CARE`, and inferring the load
op from "was a clear colour given" can never reach it.

It is also the whole of `discards`, which is what lets a barrier into the target
come from `UNDEFINED` and drop the read access bit. RPS derives the same thing
from `DISCARD_DATA_BEFORE` (`rps_vk_runtime_backend.cpp:143`).
"""
abstract type LoadOp end

struct KeepOp <: LoadOp end
struct DiscardOp <: LoadOp end
struct Clear{T} <: LoadOp
    value::T
end

"""Load what is there. What a bare target means."""
const Keep = KeepOp()

"""Do not load. The pass is asserting it writes every pixel."""
const Discard = DiscardOp()

"""Does this load op throw away the previous contents?"""
discards(::KeepOp) = false
discards(::DiscardOp) = true
discards(::Clear) = true

function Surface end
function Attribute end
"""
    draw!(pass, pipeline, args, count; frag_args = ())

Record a draw. `args` are the vertex shader's, `count` is how many vertices it
runs for, and `frag_args` are the fragment shader's.

A pipeline has one push constant range, so a draw's arguments belong to one
stage and giving them to both is an error rather than a silently wrong layout.
Which stage is the one that reads them: a mesh pass puts its buffers on the
vertex shader and hands the fragment shader varyings, and a fullscreen pass does
the reverse — its vertex shader makes a triangle out of `vertex_index()` and
takes nothing, and its fragment shader reads a g-buffer per pixel.

    shade(inputs, albedo, depth, invvp::Mat4f, w::Int32, h::Int32) = ...

    render!(g, "light", screen => Discard) do p
        draw!(p, LIGHTING, (), 3;
              frag_args = (use(p, albedo; read = true), use(p, depth; read = true),
                           inverse_vp, Int32(w), Int32(h)))
    end

A fragment shader takes its varyings first and these after, which is why `shade`
above begins with `inputs` even though a fullscreen pass has no varyings to read.

`use` rather than `Attribute` for the buffers, because `Attribute` is a vertex
binding and this is a shader read in the fragment stage; the barrier follows
from that.
"""
function draw! end

"""
    dispatch!(pass, kernel, args, ndrange; group = nothing)

Launch a kernel as part of a pass, so what it reads and writes is what orders it
against everything else.

`group` is the workgroup size, and it is worth giving one whenever `ndrange` is
two-dimensional. The default partitions an ndrange along its first axis, which
for anything image-shaped means a workgroup is one long row: a pass that reads a
row-major g-buffer and writes a column-major target then has one of the two
uncoalesced, and it costs about nine times the square arrangement — 1.06 ms
against 0.12 ms at 1280x800 on the machine this was measured on.

    dispatch!(p, shade!, args, (w, h); group = (16, 16))

`nothing` means the backend picks, which is right for a one-dimensional ndrange
and is what everything that does not say otherwise gets.
"""
function dispatch! end
function render! end

"""
    Update(graph, buf) -> ref

Reserve a position in the schedule where `buf` may be rewritten, and return a
callable that supplies the data.

    ref = Update(g, positions)
    on(obs) do new_data
        ref(new_data)          # a store. any thread, any moment.
    end

`ref(x)` retains `x` and marks the resource dirty. It does not copy, allocate or
touch the GPU, because it runs on whatever task set the observable and there is
no command buffer open there. The write happens at the reserved position during
`run!`, and the barriers around it are derived from `CopyDst` like any other
usage.

Firing twice before a frame replaces the reference; the older array is dropped.
If the caller means to mutate `x` before the next frame, they pass `copy(x)` —
there is no keyword for it, because `copy` says the same thing where a profile
can see it.
"""
function Update end

"""
    newpass(graph, name, kind) -> pass
    handle(graph, pass)        -> what a pass block is given
    dispatches(pass)           -> where its recorded work goes
    passes(graph)              -> the graph's passes, in declaration order

The four things a backend has to say about passes, so that `custom!` and
`compute!` do not have to exist twice.

They had become identical: make a pass, push it, hand a block the pass handle,
keep what it returns. Only the pass TYPE differed — `Pass(name, :custom)` against
`HostPass(name)` — which is what these hooks are for. A backend that represents
passes differently still says so in one place instead of reimplementing the
recording protocol around it.
"""
function newpass end
function handle end
function dispatches end

"""
    custombody(body) -> body

Check that a `custom!` block returned the callable it is contracted to.

The contract, not a mechanism: what a backend does with the body differs, that it
must BE one does not. Both extensions carried this test and its wording verbatim,
and an error message that exists twice is one that will eventually say two
different things about one rule.
"""
function custombody(body)
    applicable(body) || throw(ArgumentError(
        "custom!: the block has to return a zero-argument callable — it is what " *
        "runs at record time. Declare the uses, then return the work."))
    return body
end

"""
    checklive(plan, regions, ntransients)

Refuse to run a plan whose regions have been given back.

`free!` deregisters a plan from its arenas, and the bytes go once the last tenant
does. What is left behind still *looks* runnable — the steps are baked, the
storage handles are populated — so it would record against memory the pool has
handed to somebody else, and produce a plausible wrong answer rather than an
error. A plan that declared transients and holds no regions is exactly that
state, which is why this needs no flag to track.
"""
function checklive(plan, regions, ntransients::Integer)
    (ntransients > 0 && isempty(regions)) && throw(ArgumentError(
        "this plan has been freed — `free!` gave its arenas back, and running it " *
        "now would record against memory the pool may have handed to another " *
        "plan. Build a new plan from the graph."))
    return plan
end

"""
Stable small integers for the resources a graph's passes name.

Usages are recorded as `id => Usage` pairs, and the phases compare ids rather
than resources — so this table is what makes "the same bytes" a question the
scheduler can answer without knowing what a resource IS. `by_id` is the way back,
for the barrier that has to ask what it is scoping to.

In core because both backends kept one and neither kept it differently: an
`IdDict` for identity, a counter that only grows, first-come numbering. The Host
graph did not keep `by_id` and so could not answer the reverse question at all,
which is a difference in capability rather than in design.
"""
struct IdTable
    ids::IdDict{Any,Int}
    by_id::Dict{Int,Any}
end
IdTable() = IdTable(IdDict{Any,Int}(), Dict{Int,Any}())

"""The id `r` is known by, assigning one if this is the first time it is named."""
resourceid(t::IdTable, r) = get!(t.ids, r) do
    id = length(t.ids) + 1
    t.by_id[id] = r
    id
end

"""What an id names. The inverse of [`resourceid`](@ref)."""
byid(t::IdTable, id::Integer) = t.by_id[id]
Base.haskey(t::IdTable, r) = haskey(t.ids, r)

"""
The id `r` already has, erroring if it has none.

Distinct from [`resourceid`](@ref), which ASSIGNS one — asking what a resource is
called and declaring that it is named are different questions, and a test that
wants the first should not silently get the second and a fresh id for a resource
no pass ever touched.
"""
Base.getindex(t::IdTable, r) = t.ids[r]
Base.get(t::IdTable, r, default) = get(t.ids, r, default)
Base.length(t::IdTable) = length(t.ids)

"""
    registerupdate!(refs, usages, id, buf, range) -> UpdateRef

Reserve `id` as a `CopyDst` of the update pass — once, however many refs name it
— and hand back the ref that writes it.

The "once" is the part worth sharing: several `Update`s on one resource are one
hazard, and a usage list that repeats it makes the barrier phase derive the same
dependency twice. Both extensions had this test spelled the same way.
"""
function registerupdate!(refs, usages, id, buf, range)
    any(u -> u.first == id, usages) || push!(usages, id => CopyDst)
    r = UpdateRef(buf, range)
    push!(refs, r)
    return r
end

"""
What [`Update`](@ref) hands back.

Calling it stores a reference and nothing else: it runs on whatever task set the
observable, where there is no command buffer open and no guarantee the device is
done with the resource. `pending` is swapped atomically for that reason — an
observable can fire while `run!` is reading it.

In core because both backends had this verbatim, down to the `@atomic`, and two
copies of a lock-free protocol drift in exactly the place nobody looks. A backend
supplies where the write goes and how, not when it is safe to read the box.
"""
mutable struct UpdateRef
    resource::Any
    range::Union{Nothing,UnitRange{Int}}
    @atomic pending::Any
end
UpdateRef(resource, range = nothing) = UpdateRef(resource, range, nothing)

(r::UpdateRef)(data) = (@atomic r.pending = data; nothing)

"""Whether any of these refs has data waiting.

What an update pass asks before emitting its barriers: a pass whose refs are all
clean writes nothing, and ordering against a write that did not happen is cost
without a hazard."""
anypending(rs) = any(r -> (@atomic r.pending) !== nothing, rs)

"""
    applyupdates!(f, refs)

Hand each waiting write to `f(ref, data)` and consume it.

Consumed, so a ref that is not fired again writes nothing next run — the
difference between "this value changed" and "this value exists". Reading and
clearing are both here so a backend cannot implement half of it.
"""
function applyupdates!(f, rs)
    for r in rs
        data = @atomic r.pending
        data === nothing && continue
        f(r, data)
        @atomic r.pending = nothing
    end
    return nothing
end

"""
    copy!(graph, name, dst, src)

A copy as a graph pass, so its layouts and ordering are derived rather than
stated.
"""
function copy! end
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

function run! end
function npipelines end

"""
    free!(plan)

Give the plan's pool regions back. Explicit, and never a finalizer: the Block
owns the memory, the Pool owns the Block, and a plan that is dropped without
this leaks its regions until the pool is trimmed — a leak rather than a crash,
and the right way round.
"""
function free! end

"""
What one pass cost. `host_ms` is recording it, `gpu_ms` is running it.

The two are not parts of one total: recording happens this frame, the GPU work
happens when the queue reaches it, and adjacent passes overlap on the GPU. A sum
down the column is not the frame time.
"""
struct PassTiming
    name::String
    kind::Symbol
    host_ms::Float64
    gpu_ms::Float64
    samples::Int
end

Base.show(io::IO, t::PassTiming) =
    print(io, rpad(t.name, 16), lpad(round(t.host_ms; digits = 3), 8), " ms host",
          lpad(round(t.gpu_ms; digits = 3), 8), " ms gpu")

function Base.show(io::IO, ::MIME"text/plain", ts::Vector{PassTiming})
    println(io, rpad("pass", 16), rpad("kind", 9), lpad("host ms", 9), lpad("gpu ms", 9),
            lpad("n", 6))
    for t in ts
        println(io, rpad(t.name, 16), rpad(t.kind, 9),
                lpad(round(t.host_ms; digits = 3), 9),
                lpad(isnan(t.gpu_ms) ? "-" : round(t.gpu_ms; digits = 3), 9),
                lpad(t.samples, 6))
    end
end

"""
    timings(plan) -> Vector{PassTiming}

Per pass, what it cost: host time to record it and GPU time to run it.

Off unless the plan was built with `Plan(g; profile = true)`, because measuring
costs two timestamp writes per pass and a query pool. On, the cost is paid where
it is asked for rather than behind a global switch.

The GPU numbers are read without waiting: a frame whose timestamps are not back
yet is skipped rather than stalled for, so profiling does not change the frame
time it is measuring. Both figures are medians over the last `NSAMPLES` *samples*
kept, because a single frame's recording time is mostly whatever the OS was
doing. Samples, not frames: with frames in flight most reads find the results
not ready and are skipped, so four hundred frames may leave twenty samples a
pass. `PassTiming.samples` is how many are actually behind a number, and a small
one means the median is over very little.

    plan = Plan(g; profile = true)
    for _ in 1:120; run!(plan); end
    timings(plan)

is the whole of it — with the caveat that the first measurements out of a fresh
session are not the steady state. Pipelines are created, memory is faulted in
and the clocks are still ramping: on the twenty-pass renderer in bench the first
four hundred frames average 2.6 ms each and the next four hundred 0.95. Anything
comparing two configurations has to discard the first, or it is comparing the
order they were measured in.

`timings(plan)[i].gpu_ms` is pass `i` in scheduled order,
which is not declaration order when the scheduler reorders.
"""
function timings end

"""How many frames a profiled plan keeps, and what `timings` takes the median
over. Two seconds at 60 Hz, which is long enough to be stable and short enough to
follow a change."""
const NSAMPLES = 120

"""
    stride(x) -> Int

Elements to advance per index. Zero means one value shared by every element, so a
scalar and a per-element array take the same shader path and the same pipeline.
Nothing branches on it: the shader computes `1 + stride * (i - 1)` and with a
stride of zero every invocation reads element one.
"""
function stride end

"""Number of elements a draw covers, taken from the binding so it cannot disagree."""
function count end

"""
Transient resources. Constructing one against a graph or a pass means the same
thing, because liveness derives the interval from use either way.
"""
module Transient
function Buffer end
function Image end
end
