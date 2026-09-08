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
    remakeimage!(device, t)

Replace `t`'s driver image with one of `t`'s current size, and update `t.req`.
The device is the graph's, passed rather than looked up: an image belongs to the
device that made it, and the process default is not necessarily that device.

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
    refit!(x) -> Bool

Resize `x` in place to match what the plan now needs, returning whether
anything changed.

`false` means the existing resource already fits and nothing was reallocated,
which is what lets a plan be re-run at a new window size without rebuilding.
"""
function refit! end

# ── The pass walk's primitives ───────────────────────────────────────────────
#
# The walk itself — what a pass is made of and in what order — is `emit!` in
# `graph/kalaunch.jl`, and it is Mantle's. What each step DOES on a device is
# answered here, once per backend. A backend that records (Vulkan) answers with
# commands into an emitter it opened; one that does not (KernelAbstractions,
# Metal) answers on the spot through the core `Immediate` emitter, and never
# defines a method below. `record_pass!(graph, queue, passplan, derived, args;
# suppress)` was the old shape, and the Vulkan backend then grew a whole second
# copy of the walk around it.

"""
    openrecording(device, plan) -> emitter or nothing

Open a command buffer for `plan` and hand back what the walk emits into.
`nothing` from a backend that has no command buffers to build: the plan is then
walked by every `execute!` instead, which is the same walk.
"""
openrecording(::Device, ::Plan) = nothing

"""
    closerecording!(emitter, plan) -> recording

Seal what the walk emitted and give back the recording `run!` will submit,
with the plan's patch table filled from where the pack landed its addresses.
"""
function closerecording! end

"""
    emithead!(emitter, plan)

Before anything the plan emits: whatever ran last on the queue may still be
reading the pool this plan is about to write, and a cross-plan hazard cannot be
derived — so a backend puts one global barrier here, always.
"""
emithead!(e, ::Plan) = nothing

"""
    emitbarriers!(emitter, passplan)

The barriers the graph derived for this pass, before its work.
"""
emitbarriers!(e, ::PassPlan) = nothing

"""
    withpredicate(f, emitter, predicate)

Run `f` with this pass's work discarded unless its `repeat!` predicate is
nonzero. `nothing` is no predicate.
"""
withpredicate(f, e, ::Nothing) = f()

"""
    emitupdate!(emitter, plan, passplan)

An update pass: land the plan's pending host stores. Whether that can happen
where the walk is or has to wait for the run that has the values is the
emitter's answer — a recording holds no store, since it would replay this run's
values for ever.
"""
function emitupdate! end

"""
    emitkernel!(emitter, f, args...; ndrange, workgroup_size)

Launch a plain Julia kernel into the emitter — what core's fused prepare
(`emitprepares!`, in `graph/kalaunch.jl`) is written with. Never reached on a
backend whose dispatches are sized on the host.
"""
function emitkernel! end

"""
    emitpreparebarrier!(emitter)

The one barrier a fused prepare is followed by: its writes, made visible to
the command processor's read of the counts and to the dispatches behind it.
"""
emitpreparebarrier!(e) = nothing

"""
    workgroupsize(compiled) -> UInt32

Invocations per workgroup of a compiled dispatch, which is what the prepare
divides a device-written count by. One for a trace, whose indirect command
counts rays rather than groups — that method is core's.
"""
function workgroupsize end

"""
    emitdispatch!(emitter, dispatch, name)

One compiled dispatch or trace of a compute pass.
"""
function emitdispatch! end

"""
    emitcopy!(emitter, plan, passplan)

A `copy!` pass: one target into one buffer.
"""
function emitcopy! end

"""
    beginrender!(emitter, plan, passplan) -> handle
    emitdraw!(emitter, handle, draw)
    endrender!(emitter, handle)

A render pass: open its attachments, each draw in order, close it. `handle` is
whatever `beginrender!` returned and means nothing to the walk.
"""
function beginrender! end
function emitdraw! end
function endrender! end

"""
    compiledraw(c, pass, draw, argoff) -> CompiledDraw
    compile_dispatch(c, dispatch, argoff, indirect) -> compiled dispatch

What a draw or a dispatch of the graph IS on this backend, compiled once at
`Pipelines` against the layout core decided: `argoff` is the argument block
the item owns, `indirect` its indirect-command slot (zero for a host-sized
dispatch). The interpreted defaults live in `graph/kalaunch.jl`: a `Launch`,
and a draw of argument size zero.
"""
function compiledraw end
function compile_dispatch end

"""
    passbarriers(c, pass) -> (images, pre, barrier)

What the `Barriers` phase derived for this pass, in the form the backend emits:
its image layout changes, the transitions it needs, and its one memory barrier.
Nothing on a backend whose order IS its synchronisation, which is the default.
"""
passbarriers(::Compile, ::Pass) = (Nothing[], Transition[], nothing)

"""
    recordsplans(device) -> Bool

Whether this backend records plans at all. `false` by default — an
interpreted backend walks every run — and it decides only whether `run!`
refuses a recordable plan that was never `record!`ed.
"""
recordsplans(::Device) = false

"""
    openrun(device, plan) -> emitter
    closerun!(device, plan, emitter) -> token
    abandonrun!(device, emitter)
    abandonframe!(device, plan)

One run, as the backend sees it. `openrun` opens whatever this run's commands
go into — the frame's image was acquired by `beforeframe!`, before the plan was
refit, so nothing about the surface changes under an opened run; `closerun!`
closes it and hands the run to the device — the recording behind it when the
plan has one — and presents; it returns the token `waitfor!(plan)` waits on.
`abandonrun!` is what happens to an opened run whose emit threw.
`abandonframe!` is what happens to a frame that failed anywhere after its image
was acquired: the image goes back to the presentation engine untouched, so the
next frame can acquire, and nothing when the frame was presented or never
acquired. The defaults are the interpreted backend's: an `Immediate` emitter,
present, no token, and nothing to hand back.
"""
openrun(dev, pl) = Immediate()
function closerun!(dev, pl, ::Immediate)
    for s in pl.graph.surfaces
        present_frame!(dev, s.win)
    end
    return nothing
end
abandonrun!(dev, ::Immediate) = nothing
abandonframe!(dev, pl) = nothing

"""
    emitinline!(emitter, region, offset, ptr, nbytes)

`nbytes` from `ptr`, carried inside the command buffer, into `region` at
`offset`. What a pointer patch is, and (step 5) a small store.
"""
function emitinline! end

"""
    collect!(profiler, device)

Read back the timestamps the last run recorded, without waiting: a frame still
in flight is skipped rather than waited for. Nothing on a backend without
timestamp queries, which is the default.
"""
collect!(::Profiler, dev) = nothing

"""
    syncbackend(device) -> Backend

The marker the `Barriers` phase asks `needs_transition` of, and the lowering
asks `stages`, `access` and `layout` of: `VulkanAPI()`, `MetalAPI()`,
`HostAPI()`. A device says which; core never guesses from its type.
"""
function syncbackend end

"""
    profiled!(f, plan, emitter, i)

Run `f`, which emits pass `i`, and sample what it cost when the plan is
profiled. The default times the host side and asks [`gpupasstime!`](@ref); a
backend with timestamp queries writes them around the pass instead.
"""
function profiled!(f, pl::Plan, e, i::Integer)
    prof = pl.profiler
    prof === nothing && return f()
    t0 = time_ns()
    r = f()
    g = gpupasstime!(pl.graph.dev)
    sample!(prof.host_ns[i], Float64(time_ns() - t0))
    isnan(g) || sample!(prof.gpu_ns[i], g)
    return r
end

# `rename!` is gone. It pointed a resource at a fresh store rather than
# overwriting the one the GPU was reading, which is the one thing a RECORDING
# cannot follow: the commands hold the address they were written with, so they
# keep reading the store the update moved away from. It was one of the two
# reasons `record!` could refuse a plan.

"""
    storebytes!(emitter, store, offset, ptr, nbytes)

`nbytes` from `ptr` into `store` — a resource's `DeviceArray` — at byte
`offset`, as a command in this run, ordered before what the run reads. Inside
the command buffer where the backend can carry the bytes, staged where it
cannot; which is the backend's constraint, not a decision of the graph's.
"""
function storebytes! end

# `nextslot!` is gone with the argument ring. It advanced to the next of
# `ARG_SLOTS` copies of a plan's arguments and waited when the host got that far
# ahead of the device — the whole of the graph's pipelining policy, and all of it
# in service of one thing: a run wrote argument bytes on the HOST while an
# earlier run's submission could still be reading them. Nothing writes those
# bytes after `record!` any more (see [`GPURef`](@ref)), so there is nothing to
# rotate and nothing to wait for. A run is one `submit!` with no host stores and
# no possibility of a stall in front of it.

"""
    waitfor!(device, token)

Block until the device has finished the work `token` covers.

It used to hand over an open batch first when the token was not out yet, and
only then wait, because a headless plan submitted when something asked it to.
Every token is out the moment it exists now — `submit!` is the only way work
reaches the device and it goes at once — so this is the wait and nothing
else, and a token nothing will signal is an error rather than a submit.
"""
function waitfor!(dev, tok)
    waitfor(dev, tok)
    return nothing
end

"""
    waitfor!(plan)

Block until the device has finished what `run!(plan)` last submitted.

This is what a host loop reading a device-written value between runs needs, and
it is not `waitidle`: it waits for one plan's last run rather than for the
device.

A KernelAbstractions `synchronize` is the wrong tool for it in both directions.
It reaches the queue behind Mantle's back, so Mantle cannot see the stall, avoid
it, or later replace it with a device-side predicate; and it waits on the
dispatch queue AND the upload queue, where the caller wanted "the thing I just
submitted".
"""
function waitfor!(pl::Plan)
    am = pl.args
    # No argument memory on a backend that passes arguments directly, and
    # `token` is 0 until the first run — a plan that has not run has nothing
    # to wait for.
    am === nothing && return nothing
    tok = argtoken(am)      # a method, not a field read: no boxing (see types.jl)
    tok == 0 && return nothing
    passed(pl.graph.dev, tok) && return nothing
    waitfor!(pl.graph.dev, tok)
    return nothing
end


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
    makeargmemory(device, passes) -> ArgMemory or nothing

The plan's argument memory, laid out once at compile so a frame only writes
arguments: every draw's and dispatch's block at the offset `Pipelines` gave it,
then one indirect command per device-sized dispatch, 256-byte aligned.

**One copy.** It was a ring, `ARG_SLOTS` deep, because a run rewrote argument
bytes on the host while an earlier run's submission could still be reading
them. Nothing rewrites these bytes after `record!` — a value that changes
between runs is a [`GPURef`](@ref), and what a dispatch is packed with is its
address — so there is nothing to race and nothing to rotate.

The layout is Mantle's. What backs it is the backend's, through [`argbytes`](@ref)
and [`indirectslot`](@ref); a backend that passes kernel arguments directly —
anything driving KernelAbstractions rather than recording a command buffer —
answers `argbytes` with `nothing` and the plan has no argument memory.
"""
function makeargmemory(dev, passes)
    args = 0
    nind = 0
    for pp in passes
        for d in pp.draws
            args += argalign(argsize(d))
        end
        for d in pp.dispatches
            args += argalign(argsize(d))
            indirectindex(d) == 0 || (nind += 1)
        end
    end
    indbase = argalign(args)
    bytes = max(argalign(indbase + nind * INDIRECT_STRIDE), 256)
    mem = argbytes(dev, bytes)
    mem === nothing && return nothing
    store, address, ptr = mem
    # One view per device-sized dispatch, built here because the offsets never
    # move and building one per record is an allocation on the recording path.
    indirect = Any[indirectslot(dev, store, indbase + (k - 1) * INDIRECT_STRIDE) for k in 1:nind]
    return ArgMemory(store, address, ptr, Ref(UInt64(0)), indirect)
end

"""
    argbytes(device, nbytes) -> (store, address, ptr) or nothing

`nbytes` of host-writable, device-addressable memory for a plan's arguments,
for as long as the plan lives: the store the plan will retire, its device
address, and the host pointer the pack writes through. `nothing` from a
backend that binds arguments directly, which is the default.
"""
argbytes(::Device, nbytes) = nothing

"""
    indirectslot(device, store, offset) -> device array of three UInt32

The indirect command `offset` bytes into `store`, as the view a prepare kernel
writes and an indirect dispatch reads.
"""
function indirectslot end

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

# ── The vocabulary, and the test that holds the line ─────────────────────────
#
# Every core function or type a backend is allowed to add a method to. This is
# the whole surface between `src/graph/` and a backend, written down so that it
# can be CHECKED: `test/vulkan/test_backend_vocabulary.jl` enumerates the core
# functions each loaded backend actually extends and fails on any name that is
# not here. Adding a name is a design decision made in this file; a backend that
# quietly grows a `record!`, a `run!` or an `execute!` of its own fails the
# suite. The names below marked with a step are the ones
# `docs/backend-independence.md` deletes; a step is done when its names are
# gone from this list and the test still passes.
const BACKEND_VOCABULARY = (
    # devices, memory, resources
    :Device, :backend, :batchqueue, :capacity, :caps, :maxalloc, :pool, :bestshape,
    :rawalloc, :rawfree, :constraintof, :mergeconstraints, :compatible, :materialize!,
    :alignment, :bufferusage, :extrausage, :imageusage, :devicearray, :deviceview,
    :upload!, :download, :devicecopy!, :resource_moved!, :arena_moved!, :release!,
    :storage, :resourcekind, :makeimage, :remakeimage!, :AdaptedAccel,
    :supports, :supports_graphics, :supports_batch_queue, :supports_rt_pipeline,
    :supportspredicate,                       # only whether fixed-size gated work can be discarded
    # the queue and its tokens
    :devices, :defaultdevice!,
    :allocate_batch_queue!, :release_batch_queue!, :submit!, :flush!, :waitidle,
    :waitfor, :waitfor!, :passed, :fence, :reset_device!,
    # sync lowering
    :access, :stages, :layout, :needs_transition, :initial_state, :initial_usage,
    :vkformat,
    # compile: the phases are core's, a backend answers these
    :syncbackend,
    :compiledraw, :compile_dispatch, :passbarriers,
    :argbytes, :indirectslot,
    :makeprofiler, :Profiler,
    # the walk's primitives
    :openrecording, :closerecording!, :emithead!, :emitbarriers!, :withpredicate,
    :emitupdate!, :emitdispatch!, :emitcopy!, :beginrender!, :emitdraw!, :endrender!,
    :profiled!, :collect!,
    :emitkernel!, :emitpreparebarrier!, :workgroupsize,
    :storebytes!,
    :recordsplans, :recycle!, :openrun, :closerun!, :abandonrun!, :abandonframe!, :emitinline!,
    :beginframe!,
    # graphics verbs, immediate and windowed
    :Framebuffer, :Window, :Surface, :Texture2D, :Sampler, :screenshot,
    :acquire_next_image!, :present_frame!, :begin_pass!, :end_pass!, :draw!,
    :draw_in_pass!, :draw_indexed_in_pass!, :draw_indirect_in_pass!, :set_viewport!,
    :use_bindings!, :bind_textures, :blit!, :transition_image!, :readback_framebuffer,
    :readback_window, :target_extent, :target_format, :target_image, :target_view,
    # ray tracing
    :pin!, :blases,
    :build_accel!, :refit_tlas!, :set_anyhit_pipeline!, :trace_rays!,
    :trace_rays_indirect!, :trace_closest_hits!, :trace_closest_hits_indirect!,
    :trace_closest_hits_anyhit!, :trace_closest_hits_anyhit_indirect!,
)
