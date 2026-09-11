# Recording a plan on Metal, which is what "baked" has to mean here.
#
# `record!` writes a plan's commands ONCE so that every `run!` hands the same thing
# to the device instead of walking the graph in Julia. The Vulkan backend does that
# by recording a `VkCommandBuffer` and re-submitting it. Metal cannot: an
# `MTLCommandBuffer` is single-use, and committing one twice is an error, not a
# replay. The replayable unit here is `MTLIndirectCommandBuffer` — commands are
# encoded into it once and an ordinary encoder runs the range with
# `executeCommandsInBuffer:` for as long as the plan lives.
#
# What a frame costs afterwards is one encoder and one `executeCommandsInBuffer` per
# SEGMENT — no dispatch-by-dispatch Julia, no KernelAbstractions, no `mtlconvert`,
# no argument re-adaptation. Before it, every run walked the passes and re-adapted
# every argument of every dispatch; for Hikari's SoA work queues that is
# `StructArrays.replace_storage` boxing a 25-field NamedTuple, per queue, per launch,
# 320 launches a frame.
#
# ── What an indirect command can and cannot do ───────────────────────────────
#
# * It binds BUFFERS, never bytes. `setBytes:` has no counterpart on a command, so
#   every argument a launch would push has to live in memory first — which is what
#   the plan's argument memory is for, and why `argbytes` below exists at all. A
#   buffer bound at index `i` and bytes pushed at index `i` are the same binding to
#   the shader, so nothing about the compiled kernel changes.
# * Its pipeline must have been built with `supportIndirectCommandBuffers`, which is
#   `mtlfunction`'s `indirect` keyword: same compiled code, same relocation table,
#   same linked functions, a pipeline a command may name.
# * Commands are CONCURRENT unless a command carries a barrier, and two `execute`
#   calls on one encoder are ordered. Both halves are pinned by
#   `Metal/test/indirect_command_buffer.jl`; getting the first wrong loses writes
#   silently.
# * Everything a command reaches is reached by ADDRESS, so nothing is resident by
#   virtue of being bound. The pack makes every buffer persistently resident once,
#   at record — and a resource under eight bytes is not covered even then, which is
#   why `checkresident` refuses one.
#
# ── The one thing a recording cannot bake: a `repeat!` gate ──────────────────
#
# A gated pass runs or does not run on a value the DEVICE writes, and the host is
# never told which. `executeCommandsInBuffer:indirectBuffer:` is the hardware answer:
# the command processor reads a `(location, length)` pair from memory when it reaches
# the call, so a one-thread kernel recorded just ahead of the gated commands writes
# `length = 0` for a discarded iteration and the real length otherwise. The plan is
# therefore recorded as SEGMENTS — maximal runs of passes sharing a predicate — and a
# frame is one `execute` per segment.
#
# ── What is recorded and what is not ─────────────────────────────────────────
#
# Compute and update passes. A render pass needs a render encoder, and an encoder
# cannot be created from inside an indirect buffer — a recorded render pass is
# `MTLIndirectRenderCommand` inside a per-frame render encoder, a second mechanism
# rather than a longer version of this one. `openrecording` answers `nothing` for a
# plan holding one and core walks it per run, exactly as before; `norecordreason`
# says which pass and why.

"""
A plan's argument memory on Metal: one shared buffer the plan owns.

Shared, not private: `emitupdate!` writes a `GPURef`'s new value through the host
pointer between runs, and the recorded commands read the same bytes by address.
"""
function Mantle.argbytes(d::MetalDevice, nbytes::Int)
    buf = MTL.MTLBuffer(d.dev, max(nbytes, 256); storage = Metal.SharedStorage)
    Metal.make_persistently_resident!(buf)
    ptr = convert(Ptr{UInt8}, MTL.contents(buf))
    return buf, UInt64(buf.gpuAddress), ptr
end

"""
Retire a plan's argument memory: keep it alive until whatever may still be reading
it has run.

Core retires the outgoing store when a refit lays the plan out again — a frame in
flight may still name those bytes, so they cannot simply be dropped. On Vulkan the
store is a pool region and the pool's timeline defers it; here it is an `MTLBuffer`
allocated outside the pool (`argbytes` above), and what defers it is the queue: as a
root of the open batch it is released when that batch's command buffer completes,
and not before.
"""
function Mantle.retire!(::Pool, buf::MTL.MTLBuffer)
    # The device's batch, not `Metal.global_queue`: the root has to be held by the
    # command buffer that actually reads these bytes, and that is the one Mantle
    # submits to. `Device(MetalAPI())` is the process device this pool belongs to.
    Metal.record_operation!(batchqueue(Device(MetalAPI())), buf)
    return nothing
end

"""
One argument of a recorded dispatch: where its bytes live and which slot binds them.

`offset` is into the plan's argument memory and `index` is the kernel's binding slot,
both fixed when the dispatch compiles. A non-`nothing` `buffer` means the argument IS
a buffer — a top-level `MTLBuffer` or an `MtlPtr` — and binds itself rather than a
copy of its bytes.
"""
struct RecordedArg
    index::Int
    offset::Int                       # into the plan's argument memory
    nbytes::Int
    buffer::Union{Nothing,MTL.MTLBuffer}
    bufoffset::Int
end

"""
A dispatch of the graph, compiled for recording.

Everything a command needs is settled here: the pipeline, the bytes each argument
slot binds, the grid, and the adapted values the pack writes. `argsize` is what core
lays the plan's argument memory out with, and `adapted` is what `pack_recorded!`
writes into it at `record!` — held rather than re-derived, because deriving it is
the per-launch `mtlconvert` this whole path exists to delete.
"""
struct MetalRecordedDispatch{A<:Tuple,R<:Tuple}
    kernel::Metal.HostKernel
    args::Vector{RecordedArg}
    adapted::A
    # The same arguments BEFORE `mtlconvert`, for an interpreted run of this
    # dispatch — see the call operator below for why that difference matters.
    raw::R
    state::Metal.KernelState
    groups::Int
    nthreads::Int
    name::String
    argoff::Int
    argsize::Int
end

"""
Run this dispatch now, which is what a plan that was never recorded does with it.

A plan this backend cannot record — one that draws — is walked per frame, and its
compute passes are these. The kernel, the grid and the iteration space are the
compile's; only the encoding is per frame, so this is still less work than the
interpreted `Launch` it replaced, which re-entered KernelAbstractions and rebuilt
the context every time.

The arguments are the RAW ones, and that is not an oversight. `encode_arguments!`
converts each with the encoder in hand, and converting an `MtlArray` that way is
what calls `useResource` on it — which is not only about residency: it is how
Metal's hazard tracking learns that this dispatch touched those bytes. Handing it
values that were already converted skips the declaration, and the driver, seeing no
hazard, is free to overlap the command buffer behind this one. Measured: a render
pass whose indirect count this dispatch had just written read the OLD count, in
seven runs out of eight, and reading it correctly once is what made it look like
something else.
"""
function (d::MetalRecordedDispatch)()
    d.kernel(d.adapted[1], d.raw...; groups = d.groups, threads = d.nthreads)
    return nothing
end

Mantle.argsize(d::MetalRecordedDispatch) = d.argsize
# Zero, and `devicesized` false, for a dispatch over a `DeviceRange` as much as for
# one sized on the host: this backend records the range's CEILING and lets the
# kernel's own bounds check discard the surplus threads, so there is no indirect
# command for a prepare to write and no prepare to emit. See `recordedrange`.
Mantle.indirectindex(::MetalRecordedDispatch) = 0
Mantle.devicesized(::MetalRecordedDispatch) = false

"""
An entry-point name Metal will take: the kernel's own name with everything that is
not an identifier character replaced, and the argument offset behind it to keep two
dispatches of one kernel apart. `repeat!/gate-1` is a pass name, and a metallib
function called that does not link.
"""
function icb_name(name::AbstractString, i::Int)
    cleaned = map(c -> (isletter(c) || isdigit(c) || c == '_') ? c : '_', String(name))
    return "mantle_" * cleaned * "_" * string(i)
end

"""
    recorded_args(adapted, argoff) -> (args, nbytes)

Which binding slot each argument takes and where its bytes go.

The same walk `encode_arguments!` does when it pushes a launch's arguments: a ghost
type takes no slot, a buffer binds itself, and everything else is bytes. Done here,
once, so that recording is a copy into the plan's memory and running is neither.
"""
function recorded_args(adapted::Tuple, argoff::Int)
    out = RecordedArg[]
    off = argoff
    idx = 1
    for a in adapted
        T = typeof(a)
        if T <: MTL.MTLBuffer
            checkresident(a)
            Metal.make_persistently_resident!(a)
            push!(out, RecordedArg(idx, 0, 0, a, 0))
        elseif T <: Metal.MtlPtr
            checkresident(a.buffer)
            Metal.make_persistently_resident!(a.buffer)
            push!(out, RecordedArg(idx, 0, 0, a.buffer, Int(a.offset)))
        elseif Metal.isghosttype(T) || Core.Compiler.isconstType(T)
            continue                       # no slot at all, as at launch
        else
            n = sizeof(T)
            push!(out, RecordedArg(idx, off, n, nothing, 0))
            off += Mantle.argalign(n)
        end
        idx += 1
    end
    return out, off - argoff
end

"""
Refuse a resource an indirect command could not reach.

A buffer of fewer than eight bytes is not made resident by `useResource` for a
command replayed out of an indirect buffer: its reads come back as zero and its
writes are dropped, with no error anywhere. The same buffer is fine for an ordinary
dispatch, which is why nothing else in this backend has to care, and why this is
worth a sentence at record rather than a wrong number at run. Pinned by
`Metal/test/indirect_command_buffer.jl`.
"""
function checkresident(buf::MTL.MTLBuffer)
    buf.length >= 8 && return nothing
    throw(ArgumentError(
        "record!: a $(buf.length)-byte buffer is bound into a recorded command. " *
        "A resource an indirect command reaches by address is not made resident " *
        "below eight bytes: it reads as zero and its writes are dropped. Allocate " *
        "it through the device's pool, whose blocks are megabytes."))
end

"""Write one dispatch's argument bytes into the plan's memory, once."""
function pack_recorded!(ptr::Ptr{UInt8}, args::Vector{RecordedArg}, adapted::Tuple)
    i = 1
    for a in adapted
        T = typeof(a)
        (Metal.isghosttype(T) || Core.Compiler.isconstType(T)) && continue
        rec = args[i]
        i += 1
        rec.buffer === nothing || continue        # a buffer binds itself
        r = Ref(a)
        GC.@preserve r unsafe_copyto!(ptr + rec.offset,
            convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{T}, r)), rec.nbytes)
    end
    return nothing
end

"""
The kernel state a recorded dispatch carries, built once.

`launch` builds one per launch: the malloc arena and the exception mailbox are the
device's, the relocation table is the kernel's, and the seed is fresh. A recording
holds ONE, in the plan's argument memory — so a kernel using Metal's device RNG
draws the same stream every frame. Nothing in Mantle or in Hikari does: both seed
their own generators from a sample index they advance themselves.
"""
function recorded_state(d::MetalDevice, kernel::Metal.HostKernel)
    buf, buf_addr = Metal.malloc_buffer_and_gpu_address(d.dev)
    exc, exc_addr = Metal.exception_info_buffer_and_gpu_address(d.dev)
    reloc = kernel.reloc_table
    reloc === nothing || Metal.make_persistently_resident!(reloc)
    return Metal.KernelState(rand(UInt32),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, buf_addr),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, exc_addr),
        reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device},
                    reloc === nothing ? UInt64(0) : UInt64(reloc.gpuAddress)))
end

"""
The ndrange a recorded dispatch is compiled and encoded for.

A `DeviceRange` becomes its CEILING, which is core's `dispatchrange` and the same
answer the Vulkan backend compiles against. Every kernel dispatched over one
bounds-checks itself — that is a requirement of the range, repo-wide — so
over-dispatching it is defined to be a no-op for the surplus threads. What Vulkan
does on top of that, and this does not yet, is narrow the command to the count the
device wrote; here the surplus threads exit on their own.
"""
recordedrange(nd) = Mantle.dispatchrange(nd)

"""
    Mantle.compile_dispatch(compile, dispatch, argoff, indirect)

One dispatch of the graph, compiled into a command the recorder can encode.

Everything `KernelAbstractions` would work out per launch is worked out here: the
iteration space, the grid, the argument conversion, the kernel state and the
pipeline. A frame does none of it.
"""
function Mantle.compile_dispatch(c::Mantle.Compile{<:MetalDevice}, d::Mantle.Dispatch,
                                 argoff::Int, indirect::Int)
    dev = c.graph.dev
    # A `DeviceRange` with no ceiling has no fixed size to record: the count is on
    # the device and this backend cannot yet narrow a command from there, so
    # recording it would mean dispatching `INDIRECT_CEILING` threads over a queue
    # that might hold four. Core's interpreted launch reads the count on the host
    # instead — one device sync per dispatch, which is what a plan asking for this
    # already paid, and `norecordreason` refuses to record the plan.
    d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing &&
        return Mantle.bake(c, d)
    obj = Mantle.kernelfor(d.kernel, d.group, Mantle.backend(dev))
    isempty(fieldnames(typeof(obj.f))) ||
        throw(ArgumentError("dispatch!: the kernel closes over $(fieldnames(typeof(obj.f))). " *
                            "A plan resolves its arguments once, so a captured device array " *
                            "would be the one it had when the plan was built — pass it as a " *
                            "dispatch argument instead, where a rename is followed."))
    raw = map(a -> Mantle.resolve(dev, a), d.args)
    args = map(Metal.mtlconvert, raw)
    nd = recordedrange(d.ndrange)
    ndrange, workgroupsize, iterspace, _ = KA.launch_config(obj, nd, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    entry = icb_name(string(nameof(typeof(obj.f))), argoff)
    adapted = (Metal.mtlconvert(ctx), args...)
    tt = Base.to_tuple_type(map(typeof, adapted))
    kernel = Metal.mtlfunction(obj.f, tt; name = entry, indirect = true)
    # The same autotune `MetalKernels` does per launch, against the pipeline that
    # will actually run: a kernel with a dynamic workgroup size is partitioned by
    # what its own pipeline can hold, and the context is rebuilt for the partition.
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        ws = Metal.MetalKernels.threads_to_workgroupsize(kernel.maxthreads, ndrange)
        iterspace, _ = KA.partition(obj, ndrange, ws)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
        adapted = (Metal.mtlconvert(ctx), args...)
        tt2 = Base.to_tuple_type(map(typeof, adapted))
        # The repartition changes the iteration space's VALUES, not its type — a
        # static workgroup size is not repartitioned at all. Checked rather than
        # assumed: a pipeline compiled for one context type and encoded with
        # another's bytes reads its ndrange out of the wrong fields.
        tt2 === tt || (kernel = Metal.mtlfunction(obj.f, tt2; name = entry, indirect = true))
    end
    kernel.loggingEnabled && throw(ArgumentError(
        "record!: kernel $(entry) was compiled with device-side logging, whose " *
        "launch path allocates a log buffer per launch. A recorded plan has no " *
        "per-launch path; run the plan without recording it to use the log."))
    state = recorded_state(dev, kernel)
    recargs, nbytes = recorded_args((state, obj.f, adapted...), argoff)
    return MetalRecordedDispatch(kernel, recargs, adapted, raw, state,
                                 length(KA.blocks(iterspace)),
                                 length(KA.workitems(iterspace)),
                                 entry, argoff, nbytes)
end

# ── Segments, and the emitter that builds them ───────────────────────────────

"""
A run of commands the frame replays with one call.

`slot` is which `(location, length)` pair of the recording's range buffer gates it,
or `-1` for a segment that always runs. A gated segment's commands are in the buffer
like any other; what the gate decides is how many of them the command processor is
told to run.
"""
struct MetalSegment
    first::Int          # one-based, into the indirect command buffer
    last::Int
    slot::Int           # zero-based pair index into the range buffer, or -1
end

"""
The range writer for one gated segment, and where its arguments live.

`locoff`/`lenoff` are patched when the segment CLOSES: how many commands an iteration
holds is not known until its passes have been emitted, and the writer command that
carries the number was encoded before them.
"""
mutable struct MetalRangeWriter
    kernel::Metal.HostKernel
    args::Vector{RecordedArg}
    adapted::Tuple
    state::Metal.KernelState
    slot::Int
end

"""
What `record!` walks a Metal plan with: one indirect command buffer, a cursor into
it, and the segments the frame will replay.
"""
mutable struct MetalRecorder
    dev::MetalDevice
    plan::Mantle.Plan
    icb::MTL.MTLIndirectCommandBuffer
    ncommands::Int
    cursor::Int
    barrier::Bool
    segments::Vector{MetalSegment}
    # The segment being built: where it starts, what gates it, and which range slot
    # that gate writes.
    first::Int
    pred::Any
    slot::Int
    # The range writers' own arguments — the plan's argument memory was sized from
    # the plan's dispatches, and these are not among them.
    aux::MTL.MTLBuffer
    auxptr::Ptr{UInt8}
    auxcursor::Int
    ranges::Metal.MtlVector{UInt32}
    writers::Vector{MetalRangeWriter}
    # The pass about to be emitted, stashed by `emitbarriers!` — which is the only
    # hook that sees one before `withpredicate` decides what segment it belongs to.
    pass::Any
    nwriters::Int       # how many gated runs the plan holds, counted at open
    # Set when a gate is absorbed into the open iteration. Every iteration of one
    # `repeat!` names the SAME one-slot flag, so the predicate alone cannot tell the
    # next iteration from the rest of this one — the gate that went by can.
    newiter::Bool
end

"""What a recorded frame replays, and everything it must keep alive to do so."""
struct MetalRecording
    icb::MTL.MTLIndirectCommandBuffer
    segments::Vector{MetalSegment}
    rangebuf::MTL.MTLBuffer
    rangeoff::Int
    ranges::Metal.MtlVector{UInt32}
    aux::MTL.MTLBuffer
    argstore::MTL.MTLBuffer
    writers::Vector{MetalRangeWriter}
    ncommands::Int
    # What the replay declares to its encoder, and the pool's block GENERATION it
    # was built at — see `replayresources!`.
    resources::Vector{MTL.MTLBuffer}
    nblocks::Base.RefValue{Int}
end

# Nothing to hand back: the indirect command buffer, its range buffer and the
# auxiliary arguments are ObjectiveC objects the recording owns, and dropping the
# recording drops them. The plan's argument memory is the plan's and outlives it.
Mantle.release!(::MetalRecording) = nothing

"""
Whether this plan is one this backend walks per run instead of recording.

A render or a copy pass needs an encoder of a kind an indirect command buffer cannot
hold, and that is not a failure: core's answer to `openrecording === nothing` is to
walk the plan, which is what those plans always did. Everything ELSE that stops a
recording throws instead — the caller asked for one and is entitled to know why.
"""
walkedplan(pl::Mantle.Plan) =
    any(pp -> !(pp.pass.kind === :compute || pp.pass.kind === :update), pl.passes)

"""Why a plan this backend would otherwise record cannot be, or `nothing`."""
function norecordreason(pl::Mantle.Plan)
    for pp in pl.passes
        for d in pp.dispatches
            d isa MetalRecordedDispatch || return "pass \"$(pp.pass.name)\" holds a " *
                "dispatch over a `DeviceRange` with no `max`. Give the range a " *
                "ceiling — `DeviceRange(count; max = capacity)` — so the command " *
                "can be recorded against it; without one the count is only on the " *
                "device and a recorded command cannot be narrowed to it yet."
        end
    end
    return nothing
end

"""How many commands the plan needs, and how many of them are range writers."""
function planshape(pl::Mantle.Plan)
    ndispatch = 0
    nwriters = 0
    prev = nothing
    for pp in pl.passes
        pp.pass.kind === :update && continue
        ndispatch += length(pp.dispatches)
        pred = pp.pass.predicate
        pred === nothing || samepredicate(pred, prev) || (nwriters += 1)
        prev = pred
    end
    return ndispatch, nwriters
end

"""
Whether two passes are gated by the same thing, which is what puts them in one
segment. Identity of the flag AND the index: `repeat!` gives every iteration the same
one-slot flag, and what separates two iterations is the ungated gate pass between
them, not the flag.
"""
samepredicate(a::Tuple{Any,Int}, b::Tuple{Any,Int}) = a[1] === b[1] && a[2] == b[2]
samepredicate(::Tuple{Any,Int}, ::Nothing) = false
samepredicate(::Nothing, ::Any) = false

"""
    Mantle.openrecording(dev, plan) -> MetalRecorder or nothing

The indirect command buffer the plan's commands go into, sized for them.
"""
function Mantle.openrecording(d::MetalDevice, pl::Mantle.Plan)
    walkedplan(pl) && return nothing
    let why = norecordreason(pl)
        why === nothing || throw(ArgumentError("record!: " * why))
    end
    Metal.can_use_residency_sets(d.dev) || throw(ArgumentError(
        "record!: this device has no residency sets, and every buffer a recorded " *
        "command reaches is reached by address — without one nothing it reads is " *
        "resident. Run the plan without recording it."))
    ndispatch, nwriters = planshape(pl)
    ndispatch > 0 || throw(ArgumentError("record!: the plan has no dispatches to record."))
    nslots = maximum(pp -> maximum(d -> maxslot(d), pp.dispatches; init = 0),
                     pl.passes; init = 0)
    # The range writers bind more than most dispatches do, and the descriptor is
    # what every command in the buffer is bound by — a command binding past the
    # count it declared is a driver error rather than a Julia one, and the first
    # symptom is a command that quietly does nothing.
    nwriters == 0 || (nslots = max(nslots, RANGE_WRITER_SLOTS))
    nslots <= MAX_KERNEL_BUFFERS || throw(ArgumentError(
        "record!: a dispatch binds $(nslots) buffers and an indirect command may " *
        "bind $(MAX_KERNEL_BUFFERS). Pass fewer arguments, or group them in a struct."))
    # …plus the range reset, which every gated plan opens with (`emithead!`).
    ncommands = ndispatch + nwriters + (nwriters > 0 ? 1 : 0)
    # `ray_tracing`, unconditionally: a command whose kernel traces — an inline
    # ray query against a hardware acceleration structure — is refused by a buffer
    # that did not declare it, and the refusal is a MISS rather than an error.
    # Every ray in a scene rendered out of an undeclared buffer comes back empty
    # and the image is black. Which kernels trace is not a question this can ask:
    # traversal is a call inside the shader, not a property of the dispatch.
    desc = MTL.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = max(nslots, 1),
                                                  ray_tracing = true)
    # SHARED, because the HOST encodes these commands. Metal's API validation
    # refuses CPU access to a `Private` indirect command buffer outright
    # ("CPU access for MTLIndirectCommandBuffer with MTLResourceStorageModePrivate
    # storage mode is disallowed"), so the default storage made this backend
    # un-runnable under `MTL_DEBUG_LAYER=1` — which is the one place a silent
    # residency or hazard mistake gets reported. On unified memory the mode costs
    # nothing; the buffer is written once at record time and read by the command
    # processor every frame either way.
    icb = MTL.MTLIndirectCommandBuffer(d.dev, desc, ncommands;
                                       storage = MTL.MTLResourceStorageModeShared)
    # Resident for the life of the recording. The legacy encoder learns about the
    # buffer from `executeCommandsInBuffer:` itself; a submission path with no
    # `useResource` at all does not, and a command processor reading an
    # unmapped indirect command buffer faults the queue rather than erroring.
    Metal.make_persistently_resident!(icb, d.dev)
    # One 256-byte slot per range-writer argument, plus its kernel state. Sized
    # generously and once: this is record-time memory, not a frame's.
    # One 256-byte slot per argument: the head's range reset takes three, and each
    # range writer its kernel state plus six. Sized generously and once — this is
    # record-time memory, not a frame's.
    aux = MTL.MTLBuffer(d.dev, max(256 * (4 + 7 * nwriters), 256);
                        storage = Metal.SharedStorage)
    Metal.make_persistently_resident!(aux)
    ranges = Metal.MtlVector{UInt32}(undef, max(2 * nwriters, 2))
    fill!(ranges, UInt32(0))
    return MetalRecorder(d, pl, icb, ncommands, 0, false, MetalSegment[], 1, nothing, -1,
                         aux, convert(Ptr{UInt8}, MTL.contents(aux)), 0,
                         ranges, MetalRangeWriter[], nothing, nwriters, false)
end

"""How many buffer slots a dispatch's arguments take, which is what the descriptor
has to declare."""
maxslot(d::MetalRecordedDispatch) = isempty(d.args) ? 0 : maximum(a -> a.index, d.args)

"""Metal's own limit on how many buffers one command may bind."""
const MAX_KERNEL_BUFFERS = 31

"""
How many buffer slots a range writer binds: the kernel state, then one for each
argument of `metal_write_range!` — the range, the predicate, its index, the slot
base, the location and the length. Six arguments and the state; the function itself
is a ghost and takes none.
"""
const RANGE_WRITER_SLOTS = 7

# ── The gate: what a discarded iteration does to a recording ──────────────────

"""
One thread, zeroing every execution range before the frame writes any of them.

The reset is what makes absorbing a gate into the iteration before it SAFE: a
discarded iteration does not run the gate that would have written the next range,
so the next range has to already say "run nothing". Without this it would say
whatever the previous frame left there.
"""
function metal_reset_ranges!(range, n::Int32)
    @inbounds for k in Int32(1):n
        range[k] = UInt32(0)
    end
    return nothing
end

"""
One thread, writing the execution range the command processor reads next.

`base` is where this segment's pair lives in the shared range buffer, `loc` the
ZERO-BASED first command of the segment and `len` how many it holds. A closed gate
writes a length of zero, and zero commands run — the only way a recorded plan can
discard work, since the host never learns the gate's value.
"""
function metal_write_range!(range, pred, i::Int32, base::Int32, loc::UInt32, len::UInt32)
    @inbounds go = pred[i + Int32(1)].go != UInt32(0)
    @inbounds range[base + Int32(1)] = loc
    @inbounds range[base + Int32(2)] = go ? len : UInt32(0)
    return nothing
end

"""
    Mantle.emitdispatch!(recorder, dispatch, name)

Encode one dispatch as the next command of the recording.
"""
function Mantle.emitdispatch!(e::MetalRecorder, d::MetalRecordedDispatch,
                              name::AbstractString)
    am = e.plan.args
    am === nothing && throw(ArgumentError(
        "record!: the plan has no argument memory, so a recorded command has " *
        "nowhere to bind its arguments from."))
    pack_recorded!(am.ptr, d.args, (d.state, d.kernel.f, d.adapted...))
    encode!(e, d.kernel.pipeline, d.args, am.store,
            MTL.MTLSize(d.groups), MTL.MTLSize(d.nthreads))
    return nothing
end

"""
Encode a command: its pipeline, one buffer per argument slot, its grid, and the
barrier the pass boundary asked for.

A byte argument binds a slice of `store` at the offset the compile gave it; a buffer
argument binds itself. `e.barrier` is set by `emitbarriers!` and consumed here, so
the first command of a pass that needs one carries it and the rest of the pass runs
concurrently.
"""
function encode!(e::MetalRecorder, pipeline::MTL.MTLComputePipelineState,
                 args::Vector{RecordedArg}, store::MTL.MTLBuffer,
                 groups::MTL.MTLSize, threads::MTL.MTLSize)
    e.cursor += 1
    e.cursor <= e.ncommands || error("record!: the recording is longer than the " *
                                     "$(e.ncommands) commands it was sized for")
    cmd = MTL.indirect_compute_command(e.icb, e.cursor)
    MTL.set_pipeline!(cmd, pipeline)
    for a in args
        buf = a.buffer === nothing ? store : a.buffer
        off = a.buffer === nothing ? a.offset : a.bufoffset
        MTL.set_kernel_buffer!(cmd, buf, off, a.index)
    end
    MTL.dispatch_threadgroups!(cmd, groups, threads)
    # The bit, on BOTH generations. MTL4 does no automatic hazard tracking, which
    # made it look as though it must ignore a command's barrier bit too — it does
    # not: 40 chained dispatches replayed from one `executeCommandsInBuffer:` come
    # out at 40 on an MTL4 encoder at every size tried up to eight million
    # elements. The run that read 10 of 40 was an upload racing the replay from
    # the other queue, not a barrier being dropped, and splitting the execute at
    # every barrier to "fix" it cost 79 sends where one did.
    if e.barrier
        MTL.set_barrier!(cmd)
        e.barrier = false
    end
    return cmd
end

# ── The walk ─────────────────────────────────────────────────────────────────

"""
    Mantle.passbarriers(compile, pass) -> (images, transitions, barrier)

The hazards the `Barriers` phase derived for this pass, kept for the recorder.

Core's default throws them away, which was right while every pass was its own
command buffer: commit order ordered them and nothing had to be emitted. Inside an
indirect command buffer there is no commit order — see the head of `ka.jl` — so a
recorded pass needs to know whether anything before it wrote what it reads.

`Nothing[]` for the images and `nothing` for the memory barrier: a Metal texture has
no layout to transition and there is no object to build, so the whole answer is the
list, and what the recorder does with it is set one bit.
"""
Mantle.passbarriers(c::Mantle.Compile{<:MetalDevice}, p::Mantle.Pass) =
    (Nothing[], c.prepass[p], nothing)

"""
A barrier the graph derived, as the flag the next command carries.

An indirect command with a barrier waits for every command before it in the same
replay; without one they may all run at once, which is what makes a pass's
independent dispatches independent. Consumed by the first command emitted after it,
so a pass that needs one gets it on its first command and the rest of the pass still
runs concurrently.
"""
function Mantle.emitbarriers!(e::MetalRecorder, pp::Mantle.PassPlan)
    # Stashed here because this is the only hook that sees the PASS before
    # `withpredicate` has to decide which segment its work belongs to, and that
    # decision reads what the pass writes — see `gatesopen`.
    e.pass = pp
    isempty(pp.pre) || (e.barrier = true)
    return nothing
end

# A recording holds no host store: it would replay this run's values for ever. Core
# lands them in the run's own submission instead (`emitupdates!`).
Mantle.emitupdate!(::MetalRecorder, ::Mantle.Plan, ::Mantle.PassPlan) = nothing

"""
Open a segment for this pass's predicate, closing the one before it.

Consecutive passes with the same gate share a segment; a change of gate — including
to and from no gate at all — ends one. A gated segment is preceded by its range
writer, which is encoded into the segment BEFORE it so that it runs before the
command processor reads what it wrote.
"""
function Mantle.withpredicate(f, e::MetalRecorder, pred::Tuple{Any,Int})
    if !samepredicate(pred, e.pred) || e.newiter
        # The writer belongs to the segment that is open now (the one holding the
        # gate dispatch), and must wait for it: a barrier, then close, then the
        # gated segment starts after it.
        e.barrier = true
        emitwriter!(e, pred)
        closesegment!(e)
        e.pred = pred
        e.slot = length(e.writers) - 1
        e.newiter = false
    end
    f()
    return nothing
end

"""
Whether the pass about to be emitted is the GATE of the segment that is open.

A `repeat!` gate is an ungated pass that writes the flag the next iteration reads,
and it is the only ungated work that may be absorbed into the iteration before it:
if that iteration is discarded the gate does not run, its range stays zero from
`emithead!`'s reset, and the next iteration is discarded too — which is the right
answer, because a discarded iteration writes nothing the gate reads and so the gate
would have said the same thing. Once closed, a `repeat!` loop stays closed.

Asked of what the pass WRITES rather than of its name: `repeat!` declares
`use(p, pred; write = true)` on its gate, and a pass that happens to be scheduled
between two iterations without touching the flag is real work and ends the segment.
"""
function gatesopen(e::MetalRecorder)
    e.pred === nothing && return false
    pp = e.pass
    pp === nothing && return false
    pid = Mantle.resourceid(e.plan.graph, e.pred[1])
    for (id, u) in pp.pass.usages
        id == pid && Mantle.writes(u) && return true
    end
    return false
end

function Mantle.withpredicate(f, e::MetalRecorder, ::Nothing)
    # Absorbed rather than ending the segment: this is what makes a frame ONE
    # `executeCommandsInBuffer` per iteration instead of two, and the gate is the
    # only ungated pass it is sound for.
    if e.pred !== nothing
        gatesopen(e) ? (e.newiter = true) : closesegment!(e)
    end
    f()
    return nothing
end

"""Close the segment under construction, if it holds anything."""
function closesegment!(e::MetalRecorder)
    if e.cursor >= e.first
        push!(e.segments, MetalSegment(e.first, e.cursor, e.slot))
        # A gated segment's writer learns its length here and nowhere earlier: the
        # command that carries the number was encoded before the commands it counts.
        e.slot < 0 || patchwriter!(e, e.writers[e.slot + 1], e.first, e.cursor)
    end
    e.first = e.cursor + 1
    e.pred = nothing
    e.slot = -1
    return nothing
end

"""Encode the command that writes one gated segment's execution range."""
function emitwriter!(e::MetalRecorder, pred::Tuple{Any,Int})
    slot = length(e.writers)
    adapted = (Metal.mtlconvert(e.ranges),
               Metal.mtlconvert(Mantle.storage(pred[1])),
               Int32(pred[2]), Int32(2 * slot), UInt32(0), UInt32(0))
    tt = Base.to_tuple_type(map(typeof, adapted))
    kernel = Metal.mtlfunction(metal_write_range!, tt;
                               name = icb_name("write_range", slot), indirect = true)
    state = recorded_state(e.dev, kernel)
    args, nbytes = recorded_args((state, metal_write_range!, adapted...), e.auxcursor)
    e.auxcursor += Mantle.argalign(nbytes)
    w = MetalRangeWriter(kernel, args, adapted, state, slot)
    push!(e.writers, w)
    encode!(e, kernel.pipeline, args, e.aux, MTL.MTLSize(1), MTL.MTLSize(1))
    return w
end

"""Give a writer the range it writes, and pack its arguments."""
function patchwriter!(e::MetalRecorder, w::MetalRangeWriter, first::Int, last::Int)
    adapted = (w.adapted[1], w.adapted[2], w.adapted[3], w.adapted[4],
               UInt32(first - 1), UInt32(last - first + 1))
    w.adapted = adapted
    pack_recorded!(e.auxptr, w.args, (w.state, metal_write_range!, adapted...))
    return nothing
end

"""
    Mantle.emithead!(recorder, plan)

The one command that runs before anything the graph asked for: the range reset.

Only for a plan that has gates. Its arguments live in the recording's auxiliary
buffer like a range writer's, and it is the first command of the head segment, so
every `run!` clears the ranges before the first gate writes one.
"""
function Mantle.emithead!(e::MetalRecorder, pl::Mantle.Plan)
    e.nwriters == 0 && return nothing
    adapted = (Metal.mtlconvert(e.ranges), Int32(2 * e.nwriters))
    tt = Base.to_tuple_type(map(typeof, adapted))
    kernel = Metal.mtlfunction(metal_reset_ranges!, tt;
                               name = icb_name("reset_ranges", 0), indirect = true)
    state = recorded_state(e.dev, kernel)
    args, nbytes = recorded_args((state, metal_reset_ranges!, adapted...), e.auxcursor)
    e.auxcursor += Mantle.argalign(nbytes)
    pack_recorded!(e.auxptr, args, (state, metal_reset_ranges!, adapted...))
    encode!(e, kernel.pipeline, args, e.aux, MTL.MTLSize(1), MTL.MTLSize(1))
    return nothing
end

"""
    Mantle.closerecording!(recorder, plan) -> MetalRecording

Close the last segment and hand the plan what a frame replays.
"""
function Mantle.closerecording!(e::MetalRecorder, pl::Mantle.Plan)
    closesegment!(e)
    e.cursor == e.ncommands || error(
        "record!: $(e.cursor) commands were encoded and the buffer was sized for " *
        "$(e.ncommands); the walk and `planshape` disagree.")
    am = pl.args
    am === nothing && throw(ArgumentError(
        "record!: the plan has no argument memory to bind commands against."))
    return MetalRecording(e.icb, e.segments, e.ranges.data[], Int(e.ranges.offset),
                          e.ranges, e.aux, am.store, e.writers, e.ncommands,
                          MTL.MTLBuffer[], Ref(-1))
end

# Neither is reachable: `norecordreason` refuses a plan holding a copy or a render
# pass, so a call here means the refusal missed one rather than that a caller did
# something wrong. Both say so rather than failing later inside the driver.
Mantle.emitcopy!(::MetalRecorder, ::Mantle.Plan, pp::Mantle.PassPlan) =
    error("record!: pass \"$(pp.pass.name)\" is a copy pass and reached the recorder.")
Mantle.beginrender!(::MetalRecorder, ::Mantle.Plan, pp::Mantle.PassPlan) =
    error("record!: pass \"$(pp.pass.name)\" is a render pass and reached the recorder.")

# ── Running a recorded plan ──────────────────────────────────────────────────

# `false`, and deliberately: a plan this backend cannot record — one that draws — is
# given back unrecorded by `openrecording`, and `run!` walks it as it always did.
# Answering `true` would make `run!` refuse those instead.
Mantle.recordsplans(::MetalDevice) = false

"""
    Mantle.openrun(dev, plan) -> Immediate

The emitter for whatever this run has to say in front of its recording.

A run with pending host stores waits for the device first. The stores land through
the host pointer into memory the previous frame's commands may still be reading, and
unlike Vulkan — where an update is a command inside the submission — there is nothing
here to order them against. One synchronise per run that updates anything; a run that
updates nothing never opens this at all.
"""
function Mantle.openrun(d::MetalDevice, pl::Mantle.Plan)
    pl.recording === nothing || Metal.synchronize()
    # This device's queue is the one Metal.jl launches on, whatever task we are
    # on (`adoptqueue!`), and when to submit to it is the graph's question rather
    # than Metal.jl's op counter (`ownflush!`).
    adoptqueue!(d)
    ownflush!(d, true)
    return Mantle.Immediate()
end

"""
    Mantle.closerun!(dev, plan, emitter)

Submit the run: the recording behind whatever the front put in it, then present.
"""
Mantle.closerun!(d::MetalDevice, pl::Mantle.Plan, ::Mantle.Immediate) = submitrun!(d, pl)
# A run with nothing to say in front of its recording opens no emitter at all, and
# core hands `nothing` here. Two methods rather than a `Union`: core's own
# `closerun!(dev, pl, ::Immediate)` is as specific in its third argument as a Union
# is, so one method covering both is ambiguous with it rather than more specific.
Mantle.closerun!(d::MetalDevice, pl::Mantle.Plan, ::Nothing) = submitrun!(d, pl)

function submitrun!(d::MetalDevice, pl::Mantle.Plan)
    rec = pl.recording
    # The token `waitfor!(plan)` waits on, and it comes OUT of the submission: a
    # queue that signals an event per submit knows which value this run will
    # reach, and one that does not falls back to bumping its retirement counter.
    # A walked plan submitted nothing here, so it asks for a bare fence.
    tok = rec === nothing ? closeframe!(d) : replay!(d, rec::MetalRecording)
    for s in pl.graph.surfaces
        Mantle.present_frame!(d, s.win)
    end
    # Given back here rather than left off, so that a `@metal` launch outside a
    # run keeps the batching Metal.jl promises its own callers. A run that threw
    # leaves it held, which costs nothing: the next run takes it again, and a
    # `synchronize` flushes regardless of this setting.
    ownflush!(d, false)
    return tok
end

"""
Everything a replay declares to its encoder.

An indirect command reaches its buffers by ADDRESS, so nothing in the recording
tells Metal which resources the frame touches — and `useResource` is not only about
residency: it is how the driver's hazard tracking learns that this work read and
wrote those bytes. A resource that is resident but undeclared is one the driver
believes nobody used, so it is free to overlap the command buffer behind it with
this one. That is measured, on the path next door: a draw whose indirect count a
dispatch had just written read the OLD count until the dispatch declared the buffer
(see the call operator on `MetalRecordedDispatch`). Across FRAMES the same hazard is
the next replay starting before this one's writes land.

The pool's BLOCKS rather than the plan's resources: every graph resource is a slice
of one, a pool holds a handful of blocks where a plan holds thousands of resources,
and a block covers its tenants exactly. Rebuilt only when the pool has grown, so a
steady frame is one `useResources:` call over a vector that already exists.
"""
function replayresources!(d::MetalDevice, rec::MetalRecording)
    p = pool(d)
    # ONE comparison, and not the pool's lock: `blockgen` is bumped where a block
    # is created or destroyed, so a frame asks whether the list it cached is still
    # the list rather than counting one to find out. Counting was the first version
    # and it was the same mistake as the walk in `reclaim!` — scanning per frame for
    # a change that the one place able to cause it can simply announce.
    p.blockgen[] == rec.nblocks[] && return rec.resources
    # The generation is read UNDER the lock, with the copy: read after it, a block
    # added while this rebuilt would be stamped as already covered and the next
    # frame would not notice it.
    n = lock(p.lock) do
        empty!(rec.resources)
        for (_, blocks) in p.blocks, b in blocks
            b.memory isa MTL.MTLBuffer && push!(rec.resources, b.memory)
        end
        p.blockgen[]
    end
    # The plan's own memory and the recording's, which are this backend's rather
    # than the pool's and are reached by address exactly like everything else.
    push!(rec.resources, rec.argstore)
    push!(rec.resources, rec.aux)
    push!(rec.resources, rec.rangebuf)
    rec.nblocks[] = n
    return rec.resources
end

"""
One frame of a recorded plan: an encoder, one `execute` per segment, and a commit.

This is the whole of a baked frame's host work. Nothing here looks at a dispatch, an
argument or a kernel; what runs was decided when the plan was recorded.
"""
function replay!(d::MetalDevice, rec::MetalRecording)
    sub = opensubmit!(d, replayresources!(d, rec))
    for s in rec.segments
        executesegment!(d, sub, rec, s)
    end
    return closesubmit!(d, sub)
end
