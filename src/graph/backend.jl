# What the graph asks of a backend.
#
# Mantle owns the graph: the passes, the schedule, the liveness intervals, the
# placement, the barriers and the plan. A backend does not have a graph — it
# records and submits the one Mantle compiled. This file is the whole of what it
# has to answer, and it is short on purpose.
#
# The list was derived, not designed: of the 2,761 lines the Vulkan graph had,
# 1,232 across 152 definitions name no driver type at all, and the portable half
# reaches the driver half through exactly the functions below. Anything longer
# than this list means graph logic leaked back into a backend.
#
# Two candidates dropped out on inspection rather than becoming hooks:
#
#   * `isdepth` — whether an image is a depth target is decided by its element
#     type, which is a Julia question. It is defined in core below. Vulkan's
#     `aspect` is one line derived FROM it; it was the other way round, so the
#     portable question was being answered by a `VK_IMAGE_ASPECT_*` comparison.
#   * `aspect` itself, which is then purely the Vulkan backend's and appears
#     nowhere in Mantle.

"""
    isdepth(x) -> Bool

Whether `x` is a depth target rather than a colour one.

Decided by element type: `Float32` is depth, everything else is colour. No
backend is consulted, which is why this is a definition and not a hook — every
API agrees on what a depth image is, and only their spellings of it differ.
"""
isdepth(::Type{Float32}) = true
isdepth(::Type) = false
isdepth(x) = isdepth(typeof(x))

"""
    makeimage(device, T, width, height, srgb, usage, source) -> TransientImage

Create a render target's driver image, unbound, and wrap it in a
[`TransientImage`](@ref).

The backend builds the whole struct rather than being handed one to fill,
because five of its fields ARE driver objects and their types are what the
backend decides. It must not allocate memory for the image: what the placer
needs is the SIZE and ALIGNMENT the image will want, and answering that is the
only reason the object exists this early.

`first`/`last` start at `typemax(Int)`/`0` — the empty interval liveness
narrows.
"""
function makeimage end

"""
    remakeimage!(t)

Replace `t`'s driver image with one of `t`'s current size, and update `t.req`.

Called by [`refit!`](@ref) after a tracking target's source has changed size.
The format and usage are `t`'s own and are not recomputed: they came from the
element type, which has not changed, and re-deriving them would be a second
place for them to be decided.
"""
function remakeimage! end

"""
    target_extent(target) -> (width, height)

The pixel dimensions of something a pass renders into.

Returns a plain tuple. It used to return a `VK.Extent2D`, which put a driver
type in the graph's arithmetic — the graph compares extents to check that every
attachment in a pass agrees, and that comparison is not Vulkan's.
"""
function target_extent end

"""
    checkextents(plan)

Verify every pass in `plan` renders into attachments that agree on size.

A backend answers because it knows what its targets measure; the graph asks
because a mismatch is a graph error, not a driver one — the driver would only
report it later, as a validation message naming a framebuffer.
"""
function checkextents end

"""
    refit!(x) -> Bool

Resize `x` in place to match what the plan now needs, returning whether
anything changed.

`false` means the existing resource already fits and nothing was reallocated,
which is what lets a plan be re-run at a new window size without rebuilding.
"""
function refit! end

# `record!` is declared in `Mantle.jl` with the default that makes it meaningful
# on a backend that has no command buffers to build: the plan, unchanged. The
# per-pass half is not a hook at all any more — a backend that records emits into
# something of its own (`Emitter` on Vulkan), and what that is has no portable
# spelling. `record_pass!(graph, queue, passplan, derived, args; suppress)` was
# the old one, and four of its six arguments were the queue-shaped machinery step
# 2 deleted.

"""
    rename!(graph, queue, dst, data)

Point `dst` at `data` without copying, when the graph has proved the old
contents are dead.

The optimisation that makes an `Update` cheap: a buffer the next pass fully
overwrites does not need its previous contents moved, so the graph hands the
backend a rename instead of a copy.
"""
function rename! end

"""
    inplace!(queue, dst, data, from)

Write `data` into `dst` starting at index `from`, for the case
[`rename!`](@ref) cannot take — the destination is aliased, or partially live.
"""
function inplace! end

"""
    nextslot!(args, device) -> Int

Advance to the next argument slot, once the device is finished with what it
holds, and return which slot that is.

This is the whole of the graph's pipelining policy: `ARG_SLOTS` runs of
arguments may be in flight, and the run that would make it `ARG_SLOTS + 1`
waits. It is policy and not mechanism, which is why it is here — the backend
answers `passed`, `waitfor` and `submit!` about one opaque token and decides
none of this.

It was the backend's, as `nextslot!(am, bq)` reading `bq.timeline_sem` and
`bq.next_timeline` directly. Two things followed from that. The depth of a
Mantle plan was a Vulkan constant; and the "has it been handed over yet" case
below had to be expressed as `want > bq.next_timeline`, a comparison against a
counter that only one backend has. Getting it wrong there was not an assertion,
it was `vkWaitSemaphores` on a value nothing would ever signal — a foreign call
that never returns, so the process stops dead with no Julia frame to show.
"""
function nextslot!(am::ArgMemory, dev)
    am.slot = mod1(am.slot + 1, ARG_SLOTS)
    tok = am.slot_token[am.slot]
    # Never used, or the device is already past it.
    tok === nothing && return am.slot
    passed(dev, tok) && return am.slot
    waitfor!(dev, tok)
    return am.slot
end

"""
    waitfor!(device, token)

Block until the device has finished the work `token` covers, handing that work
over first if the host is still holding it.

`waitfor(device, token)` — no `!` — is the backend's mechanism and answers
`false` for exactly one case: the token belongs to work that has not been
submitted, so nothing will ever signal it. That happens because a plan drawing
to a surface submits every run while a headless one submits when something asks
it to, so several runs otherwise accumulate in one open batch. Waiting on those
blocks in a foreign call that never returns.

Needing the result is precisely the reason to hand the work over, so that is
what happens — but ONLY then. Submitting unconditionally reads the same and is
not: it breaks up the batch every time, and a renderer running several plans per
sample pays a submission per wrap of the argument ring. Measured at ~20% across
four RayDemo scenes.
"""
function waitfor!(dev, tok)
    waitfor(dev, tok) && return nothing
    submit!(dev)
    waitfor(dev, tok)
    return nothing
end

"""
    waitfor!(plan)

Block until the device has finished what `run!(plan)` last submitted.

This is what a host loop reading a device-written value between runs needs, and
it is not `waitidle`: it waits for one plan's last run rather than for the
device, and it knows to submit that run if it is still sitting in an open batch.

A KernelAbstractions `synchronize` is the wrong tool for it in both directions.
It reaches the queue behind Mantle's back, so Mantle cannot see the stall, avoid
it, or later replace it with a device-side predicate; and it waits on the
dispatch queue AND the upload queue, where the caller wanted "the thing I just
submitted".
"""
function waitfor!(pl::Plan)
    am = pl.args
    # No argument memory on a backend that passes arguments directly, and `slot`
    # is 0 until the first `nextslot!` — a plan that has not run has nothing to
    # wait for, and `slot_token[0]` is a `BoundsError` rather than an answer.
    (am === nothing || am.slot == 0) && return nothing
    tok = am.slot_token[am.slot]
    tok === nothing && return nothing
    passed(pl.graph.dev, tok) && return nothing
    waitfor!(pl.graph.dev, tok)
    return nothing
end

"""
    collect!(profiler, ctx)

Read back the timestamps a run recorded.

Separate from the run because it synchronises, and a caller that is not reading
timings should not pay for it.
"""
function collect! end

"""
    makeprofiler(device, passes, profile::Bool)

The plan's profiler, or `nothing` when the caller did not ask for one.

A hook rather than a plain constructor because a backend with TIMESTAMP QUERIES
builds a query pool here, which is a device object. Without one the default
below still answers: it times the host side of every pass, which is the whole
of what `run!` can see and enough to say where a frame goes.

That default matters. It used to be `nothing` unconditionally, so
`Plan(g; profile = true)` on any backend but Vulkan silently produced a plan
with no profiler and `timings` had nothing to report — a question asked and
quietly dropped.
"""
makeprofiler(dev::Device, passes, profile::Bool) =
    profile ? hostprofiler(passes) : nothing

"""
A profiler that measures the host side only.

`pool = nothing` and `period_ns = NaN` say there are no timestamps: `timings`
reports `gpu_ms` as `NaN` rather than inventing a number, and a backend that
HAS them overrides `makeprofiler` and fills `gpu_ns` itself.

What the host side measures is the time `run!` spends in a pass — recording,
binding, submitting, and any wait the backend does inside it. On a backend that
records and returns, that is not the GPU cost; it IS the frame's serial cost,
which is what you are looking at when a frame is slower than the sum of its
shaders.
"""
hostprofiler(passes) =
    Profiler(nothing, NaN, 0, [pp.pass.name for pp in passes],
             [Float64[] for _ in passes], [Float64[] for _ in passes], false)

"""
    gpupasstime!(device) -> Float64

Nanoseconds the GPU spent on the work the pass just recorded, or `NaN` if this
backend cannot say.

Called by `run!` once per pass, and ONLY while profiling. It is allowed to be
expensive — a backend without timestamp queries answers this by submitting what
was recorded and waiting for it, which serialises a frame that would otherwise
pipeline. That is the trade a profiled run makes, and it is why the number is a
per-pass cost rather than a frame time: the sum will exceed what the same frame
takes unprofiled.

`NaN` is the honest default. A backend that has neither timestamps nor a way to
bracket its submissions says so, and `timings` reports it rather than a zero
that reads as "free".
"""
gpupasstime!(::Device) = NaN


"""
    makeargmemory(device, passes)

The plan's argument memory, or `nothing`.

Laid out once at compile so a frame only writes arguments. A backend that passes
kernel arguments directly — anything driving KernelAbstractions rather than
recording a command buffer — has none, which is the default.
"""
makeargmemory(::Device, passes) = nothing

"""
    register_kernel_recorder!(recorder; name)

Register `recorder` as a backend's kernel-cache recorder.

`recorder(f, version)` runs `f` with that backend's on-disk cache recording
under `version`, and restores whatever it changed afterwards. A backend that
caches compiled kernels — Vulkan freezes SPIR-V — registers one from its
extension's `__init__`; one that does not registers nothing.

A registry and NOT dispatch, which is what this was: the backend wrote
`Mantle.with_kernel_recording(f, version) = …`, the same untyped two-argument
signature as the default below, so it OVERWROTE it rather than extending it —
fatal during precompilation, and silently the wrong hook if it had not been. The
call site has neither a device nor a backend to dispatch on, deliberately:
`@compile_workload` runs where there may be no driver at all, and asking for one
to decide how to record would defeat the point.

Nesting also answers the two-backend case, which dispatch could not have: with
Vulkan and Metal both loaded, every registered cache records, rather than one
method winning. `name` replaces an earlier registration under the same name, so
reloading an extension does not stack recorders.
"""
function register_kernel_recorder!(recorder; name::Symbol)
    filter!(e -> first(e) !== name, KERNEL_RECORDERS)
    push!(KERNEL_RECORDERS, name => recorder)
    return nothing
end

const KERNEL_RECORDERS = Pair{Symbol,Any}[]

"""
    with_kernel_recording(f, version)

Run `f` with every registered kernel cache recording under `version`.

The hook behind [`@compile_workload`](@ref). With nothing registered this is
`f()`, which is the whole behaviour on a backend that compiles kernels fresh
each session.
"""
function with_kernel_recording(f, version)
    isempty(KERNEL_RECORDERS) && return f()
    inner = f
    for (_, recorder) in KERNEL_RECORDERS
        outer = inner                       # rebound per iteration, so each
        inner = () -> recorder(outer, version)   # closure keeps its own link
    end
    return inner()
end

"""
    @compile_workload version begin … end

Precompile the enclosed workload, recording any kernels the backend compiles.

PrecompileTools' `@compile_workload` with [`with_kernel_recording`](@ref) around
it. `version` names the cache generation, so a backend that persists compiled
kernels can invalidate them without invalidating the package image.
"""
macro compile_workload(version, ex)
    return esc(quote
        $PrecompileTools.@compile_workload begin
            $with_kernel_recording($version) do
                $ex
            end
        end
    end)
end
