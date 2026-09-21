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
#     type, which is a Julia question. It is defined in core below, and Vulkan's
#     `aspect` is one line derived FROM it rather than the other way round.
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

Returns a plain tuple and not a driver type: the graph compares extents to check
that every attachment in a pass agrees, and that comparison is not Vulkan's.
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
# The walk itself — what a pass is made of and in what order — is `emitplan!` in
# `graph/kalaunch.jl`, and it is Mantle's. What each step DOES on a device is
# answered here, once per backend. A backend that records (Vulkan) answers with
# commands into an emitter it opened; one that does not (KernelAbstractions,
# Metal) answers on the spot through the core `Immediate` emitter, and never
# defines a method below.

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
    runscalls(device) -> Bool

Whether this device can run a CALL: a pass member with no ndrange, which submits
its own work rather than being launched. See the three-argument `dispatch!`.

Beside `recordsplans` above because it is the same shape of question — what this
device's RUN path can do — and the two are related without being the same. Two
shapes answer yes:

  * a backend that WALKS its plans calls the function in order;
  * a backend whose recording is a stream CAPTURE gets it for free, because the
    library's own submission lands in the capture.

A backend that builds a command buffer itself and submits only that answers no.
The call would run on the host once, at record time, submit its work somewhere
other than the buffer being built, and be missing from every replay after that —
silently. So the answer is no rather than yes-with-a-caveat, and the operation
is declared as dispatches there instead.

`false` by default: the answer is about a run path, and core cannot infer it.
"""
runscalls(::Device) = false

"""
    librarygemm(device, out, A, B, bias, epilogue) -> callable or nothing

Return a backend library call that computes `out = epilogue.(A * B .+ bias)`,
or `nothing` when this device/library cannot implement that exact combination.
The callable takes `(out, A, B, bias)`; keeping the device-owned library handle
inside it means callers pass devices explicitly and no ambient or cached device
identity is involved.

This is a capability query rather than a vendor branch.  A command-buffer
backend normally answers `nothing`; a stream-capture backend may return a
vendor-library call whose submitted kernels become part of the recording.
"""
librarygemm(::Device, out, A, B, bias, epilogue) = nothing

"""
    native_gemm_dispatch!(device, graph, out, A, B;
                          bias=nothing, epilogue=identity, name)

Declare a backend's recordable native GEMM kernel into `graph`.  Return `nothing`
when no native kernel covers the operands, otherwise return
`(; bias::Bool, epilogue::Bool)` saying which requested post-operations the
dispatch folded into its store.  DNNKernels uses those two generic facts to
declare any remaining passes; a backend does not participate in graph planning.

Unlike `librarygemm`, this path is a device dispatch and can therefore be baked
into a command-buffer recording.
"""
native_gemm_dispatch!(::Device, graph, out, A, B;
                      bias = nothing, epilogue = identity, name) = nothing
# Immediate execution carries a KernelAbstractions backend while declaration
# carries a Mantle device.  Backends may answer for either; the default is the
# same absence of a native path.
native_gemm_available(::Union{Device,KA.Backend}, ::Type, ::Type, ::Type) = false

"""
    native_batched_gemm_dispatch!(device, graph, out, A, B;
                                  transpose_a=false, transpose_b=false,
                                  alpha=1, coldiv=nothing, name)

Declare a backend's recordable strided-batched GEMM into `graph`.  The first
two axes are the matrix axes and all trailing axes form the batch; corresponding
matrix planes are multiplied without broadcasting.  `transpose_a` and
`transpose_b` apply within every plane and `alpha` scales the product.

`coldiv` divides each output COLUMN by one value, laid out `(N, batch...)` — the
epilogue an unnormalized softmax leaves for the product that follows it. A
backend that fuses it into the product's store saves a whole pass over the
output; one that cannot simply returns `false` and the caller normalizes itself.

Return `true` when the dispatch was declared and `false` otherwise.  The
operation and its optional epilogue are backend-independent; a backend only
decides whether its native recordable kernel covers the given operands.
"""
native_batched_gemm_dispatch!(::Device, graph, out, A, B;
                              transpose_a = false, transpose_b = false,
                              alpha = 1, coldiv = nothing, name) = false

"""
    native_attention_dispatch!(device, graph, out, q, k, v; scale, name)

Declare a backend's recordable FUSED attention into `graph`: `out = softmax(scale·qᵀk)·v`,
one matrix per trailing-axis batch entry, with `q`/`out` shaped `(E, Lq, batch...)` and
`k`/`v` shaped `(E, Lk, batch...)`.

Fused means the `Lq x Lk` scores are never a buffer, which is the whole point of asking:
the three-pass form writes, rewrites and rereads them, and at a few thousand tokens that
traffic is what the op costs. A backend answers `false` for the shapes its kernel does not
cover and the caller declares the three passes instead.

Any operand may be given as a STRIDED VIEW rather than as a resource: a named tuple
`(; res, dims, strides, offset)`, strides and offset in elements. A backend whose kernel
reads its operands through a leading dimension takes the view where it lies, which saves the
caller a transpose per operand — q/k/v arrive permuted from a qkv projection. One that cannot
returns `false`, and the caller materialises and asks again.

Return `true` when the dispatch was declared and `false` otherwise.
"""
native_attention_dispatch!(::Device, graph, out, q, k, v; scale, name) = false

"""
    patchable(device) -> Bool

Whether a moved address can be written INTO this backend's recording, or whether
the recording has to be thrown away and made again.

`true` by default, which is the Vulkan answer and the one the pool's move path
assumed outright: a recorded command holds a device address as eight bytes in
the plan's argument memory, so `notify_move!` queues a patch and the recording
survives. `false` is a backend whose recording is opaque — a captured HIP graph
holds its kernel arguments where only `hipGraphExecKernelNodeSetParams` could
reach them, and capture does not hand back the node handles that would need.
Such a plan is [`invalidate!`](@ref)d instead, and the next `run!` records it
again before it submits.

Asked rather than assumed for the reason [`deviceaddress`](@ref) replaced
`resource_moved!`: a backend that cannot patch and is never asked is a recording
that quietly keeps reading the old storage. The images-arena path invalidates
rather than patches, and this is the same fact as a property of the BACKEND
rather than of one arena kind.
"""
patchable(::Device) = true

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

"""
    abandonrecording!(device, emitter)

A recording that was opened and will not be closed, because the walk threw.

[`abandonrun!`](@ref)'s counterpart for `record!` rather than `run!`, and a
separate verb because the two are different objects: a run's emitter owns a
one-shot buffer the channel lends it for that submission, a recording's owns one
the PLAN holds for as long as it lives. The Vulkan backend's `abandonrun!` asserts
the first of those, so it cannot stand in for this.

A partitioned recording reaches this with pieces already closed (`recordparts!`
releases those and abandons this one), and an unpartitioned `record!` whose walk
throws would otherwise leak a command buffer.

The default is `nothing`, for the backends whose `openrecording` hands back
nothing and so never have one open. **A backend that records must answer it**,
or a failed `record!` leaks a command buffer per attempt.
"""
abandonrecording!(::Device, e) = nothing
abandonframe!(dev, pl) = nothing

"""
    emitinline!(emitter, region, offset, ptr, nbytes)

`nbytes` from `ptr`, carried inside the command buffer, into `region` at
`offset`. What a pointer patch is, and what a small host store is.
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

"""
    storebytes!(emitter, store, offset, ptr, nbytes)

`nbytes` from `ptr` into `store` — a resource's `DeviceArray` — at byte
`offset`, as a command in this run, ordered before what the run reads. Inside
the command buffer where the backend can carry the bytes, staged where it
cannot; which is the backend's constraint, not a decision of the graph's.
"""
function storebytes! end

"""
    waitfor!(device, token)

Block until the device has finished the work `token` covers.

Every token is out the moment it exists — `submit!` is the only way work
reaches the device and it goes at once — so this is the wait and nothing else,
and a token nothing will signal is an error rather than a submit.
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

That default matters: answering `nothing` unconditionally makes
`Plan(g; profile = true)` on any backend but Vulkan a plan with no profiler and
`timings` with nothing to report, which is a question asked and quietly
dropped.
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

**One copy, not a ring.** Nothing rewrites these bytes after `record!` — a
run that rewrote them on the host could race an earlier run's submission still
reading them, and a value that changes
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

"""
    devicename(device) -> String

What GPU this is, for a human: "AMD Radeon RX 7900 XTX (RADV NAVI31)".

`devices(api)` already answers this for the devices a caller could PICK; this
answers it for the one they HAVE. RayDemo's benchmark harness wanted it to name
a results file and had to read `vk_context().device_name` to get it, which is a
demo reaching past the portable API for something entirely ordinary.

For display only. Nothing should branch on it — a driver that renames itself
between releases would change behaviour, and what a caller actually wants to ask
is `caps(device)` or one of the `supports_*` predicates.
"""
function devicename end

"""
    initbackend!()

Whatever the backend this build compiled in has to do once, at `__init__`.

Declared here and answered in `src/vulkan/` or `src/metal/`, rather than written
as a body inside the `@static if` in `Mantle.jl`: a Vulkan `initbackend!` names
`vulkan_available`, `LavaBackend` and three more things only that backend
defines, and a core file naming those is what `test_core_names_no_backend.jl`
exists to catch. Metal's does one registration and no threads.
"""
function initbackend! end

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
    # what a backend does once, at load
    :initbackend!,
    # for humans: which GPU this is
    :devicename,
    # devices, memory, resources
    :Device, :backend, :kibackend, :batchqueue, :capacity, :caps, :maxalloc, :pool,
    :bestshape,
    :rawalloc, :rawfree, :constraintof, :mergeconstraints, :compatible, :materialize!,
    :alignment, :bufferusage, :extrausage, :imageusage, :devicearray, :deviceview,
    :deviceslice,
    :upload!, :download, :devicecopy!, :hostspan, :deviceaddress, :patchable, :release!,
    :indexbuffer,
    # The recording primitives a submission channel needs — 2.3. Core owns the
    # free list and when a recording may be reused; these are the driver half.
    :makerecording, :resetrecording!, :destroyrecording!, :finishrecording!, :recorder,
    # 2.2: the driver half of lifetime. Core decides WHEN a destructor may run
    # and which channel has to wait for which; `rawfree` is the destructor and
    # `stampof` is the storage the decision is written into. `hold!` and
    # `holdleaves!` are core's verbs, and a backend adds the method that says
    # what a claim means on its own closed command buffer — a reference AND the
    # driver fact "these commands name this buffer".
    :stampof, :deviceof, :hold!, :holdleaves!,
    # The hand-recorded pass — see `graphics/record.jl`. The first four are what
    # `pass!`/`draw!` lower to, and `beginrender!(::Immediate, …)` forwards to
    # the same four, so a backend writes one pass implementation and both the
    # graph's walk and a hand-recorded pass reach it.
    :compile_draw, :begin_render_pass!, :record_draw!, :end_render_pass!,
    :setviewport!, :colorimage, :depthimage, :currentimage, :blittarget,                             # Vulkan needs a usage bit at allocation, Metal does not
    :storage, :resourcekind, :makeimage, :remakeimage!, :AdaptedAccel,
    :supports, :supports_graphics, :supports_geometry_stage,
    :supports_tessellation, :supports_mesh_pipeline, :supports_batch_queue,
    :supports_rt_pipeline,
    :supportspredicate,                       # only whether fixed-size gated work can be discarded
    # the queue and its tokens
    :devices, :defaultdevice!,
    :allocate_batch_queue!, :release_batch_queue!, :submit!, :flush!, :waitidle,
    :waitfor, :waitfor!, :passed, :fence, :reset_device!, :awaitwrites,
    # sync lowering
    :access, :stages, :layout, :needs_transition, :initial_state, :initial_usage,
    # Vendor-named verbs are not in this list: `vkformat` is the Vulkan
    # backend's own and `mtlformat` is Metal's.
    # compile: the phases are core's, a backend answers these
    :syncbackend,
    :compiledraw, :compile_dispatch, :passbarriers,
    :argbytes, :indirectslot,
    # Argument packing: core owns the classification and both walks
    # (`graph/packing.jl`); a backend adds `passedas` for its OWN allocation
    # handles and never restates the rule for what takes no slot at all.
    :passedas,
    # What an `llvmcall`'s external symbol does to the pointer it is handed.
    # A declaration and not a hook with a default answer: `nothing` means
    # undeclared, which the walk refuses rather than guesses about.
    :intrinsic_usage,
    # What core asks a backend's OWN compiled dispatch — the type
    # `compile_dispatch` returned. Core answers it for its own
    # `CompiledDispatch`, `Launch` and `Call`, so a backend that reuses those
    # says nothing; one that returns its own type has to. The other three of the
    # four are listed below, once.
    :callgroup,
    :makeprofiler, :Profiler,
    # Asked of whatever `compile_dispatch` returned. DECLARED, because core
    # dispatches through them and a backend must answer: undeclared, they were
    # contract points the ratchets below could not see. They exist at all only
    # because the Metal backend defines `MetalRecordedDispatch` instead of using
    # core's `CompiledDispatch`, which already carries these three fields under
    # different names — a backend defining a graph type, which is the thing this
    # vocabulary exists to prevent. They should disappear, not grow.
    :argsize, :devicesized, :indirectindex,
    # Likewise declared rather than left implicit: core defines each of these and
    # a backend overrides it.
    :retire!, :blocksize, :copy_target!, :gpupasstime!,
    # the walk's primitives
    :openrecording, :closerecording!, :emithead!, :emitbarriers!, :withpredicate,
    :emitupdate!, :emitdispatch!, :emitcopy!, :beginrender!, :emitdraw!, :endrender!,
    :profiled!, :collect!, :argument_usage,
    :emitkernel!, :emitpreparebarrier!, :workgroupsize,
    :storebytes!,
    :recordsplans, :runscalls, :librarygemm, :native_gemm_available,
    :native_gemm_dispatch!, :native_batched_gemm_dispatch!,
    :native_attention_dispatch!,
    :openrun, :closerun!, :abandonrun!, :abandonframe!, :emitinline!,
    # Submitting ONE baked piece, and giving one back that will never be
    # submitted. Core owns the sequence a partition makes of them
    # (`RecordingParts`) and the walk that builds it.
    :submitrecording!, :abandonrecording!,
    :beginframe!,
    # graphics verbs, immediate and windowed
    :Framebuffer, :Window, :Surface, :Texture2D, :Sampler, :screenshot,
    :acquire_next_image!, :present_frame!, :begin_pass!, :end_pass!, :draw!,
    :draw_in_pass!, :draw_indexed_in_pass!, :draw_indirect_in_pass!, :set_viewport!,
    :use_bindings!, :bind_textures, :transition_image!, :readback_framebuffer,
    :readback_window, :target_extent, :target_format, :target_image, :target_view,
    # What a pass touches, read off its kernels. Core owns the walk; a backend
    # owns the SIGNATURE it walks — which interpreter compiles this kernel, what
    # the device-side type of an argument is, and what leading arguments a kernel
    # of its own has. `shadertouches` and the two draw stages are declared with no
    # method at all, so they are not core functions a backend EXTENDS. Access
    # inference itself is core; backends provide only their array/type facts and
    # compiler interpreter, with signature overrides where their launch differs.
    :argtype, :devicebuffertype, :isdevicearray, :accesscache, :kerneltouches,
    :kernelinterpreter, :kakernelaccesssignature,
    # ray tracing
    :build_accel!, :refit_tlas!, :set_anyhit_pipeline!, :trace_rays!,
    :trace_rays_indirect!, :trace_closest_hits!, :trace_closest_hits_indirect!,
    :trace_closest_hits_anyhit!, :trace_closest_hits_anyhit_indirect!,
)
