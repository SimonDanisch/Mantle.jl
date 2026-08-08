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
    copy!(graph, name, dst, src)

A copy as a graph pass, so its layouts and ordering are derived rather than
stated.
"""
function copy! end
function compute! end
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
