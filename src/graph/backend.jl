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

"""
    record!(plan, queue; derived = true, suppress = derived, updates = false)

Record `plan`'s passes into `queue`.

The graph has already decided the order, the barriers and the memory; this
writes them into whatever the backend records into. `derived` selects the
barriers the graph computed over ones declared by hand, and `suppress` elides
barriers the graph proved redundant.
"""
function record! end

"""
    record_pass!(graph, queue, passplan, derived, args; suppress = false)

Record one pass. The per-pass half of [`record!`](@ref).
"""
function record_pass! end

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
    nextslot!(args, queue)

Advance to the next argument slot for a launch.

Arguments are written into a ring of slabs so consecutive launches do not wait
on each other; the graph knows how many launches there are, the backend knows
how big a slab is.
"""
function nextslot! end

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
