# Recording a plan on Metal. An `MTLCommandBuffer` is single-use, so the
# replayable unit is an `MTLIndirectCommandBuffer`, encoded once and run each
# frame with one `executeCommandsInBuffer:` per segment. Indirect commands bind
# buffers, never bytes (arguments live in `argbytes` memory), need pipelines built
# with `indirect = true`, run concurrently unless one carries a barrier, and reach
# memory by address, so everything they touch is made resident explicitly. A
# `repeat!` gate is a device-written `(location, length)` pair that
# `executeCommandsInBuffer:indirectBuffer:` reads, so the plan is recorded as
# segments of passes sharing a predicate. Plans with render or copy passes are
# walked per run instead.

"""
A plan's argument memory: one persistently resident shared buffer the plan owns.
Shared because `emitupdate!` writes `GPURef` values through the host pointer
between runs. Returns `(buffer, gpuaddress, hostpointer)`.
"""
function Mantle.argbytes(d::MetalDevice, nbytes::Int)
    buf = MTL.MTLBuffer(d.dev, max(nbytes, 256); storage = Metal.SharedStorage)
    Metal.make_persistently_resident!(buf)
    ptr = convert(Ptr{UInt8}, MTL.contents(buf))
    return buf, UInt64(buf.gpuAddress), ptr
end

"""
A device-sized dispatch's 256-byte slot in the plan's argument memory. Words 1-3
are the grid core's prepare writes, read by an encoded frame's indirect dispatch.
An indirect command cannot read its grid from memory, so for a replayed frame the
recorder also writes the indirect command buffer's handle (words 5-6, zero when encoding),
the zero-based command index (word 7) and threads per group (word 8), which
`writegroups!` uses to rewrite the command.
"""
struct MetalIndirectSlot
    store::MTL.MTLBuffer
    offset::Int
    # The three grid words as a device array (`slotgrid`).
    grid::Metal.MtlVector{UInt32}
end

Mantle.indirectslot(d::MetalDevice, store::MTL.MTLBuffer, off::Int) =
    MetalIndirectSlot(store, off, deviceslice(d, UInt32, (3,), store, off))

Mantle.slotgrid(s::MetalIndirectSlot) = s.grid

"""The slot as a prepare kernel holds it: its address."""
struct ICBGridSlot
    ptr::Core.LLVMPtr{UInt32,Metal.AS.Device}
end

Adapt.adapt_storage(::Metal.Adaptor, s::MetalIndirectSlot) =
    ICBGridSlot(reinterpret(Core.LLVMPtr{UInt32,Metal.AS.Device},
                            s.store.gpuAddress + UInt64(s.offset)))

"""
Inside core's prepare: write the grid, and for a replayed recording rewrite the
command's grid. The rewrite keeps the recorded pipeline and buffers, and the
command processor sees it from the next `execute`, not the one that wrote it, so
`emitpreparebarrier!` closes the segment.
"""
@inline function Mantle.writegroups!(s::ICBGridSlot, groups::UInt32)
    p = s.ptr
    unsafe_store!(p, groups, 1)
    unsafe_store!(p, UInt32(1), 2)
    unsafe_store!(p, UInt32(1), 3)
    icb = unsafe_load(reinterpret(Core.LLVMPtr{UInt64,Metal.AS.Device}, p + 16))
    if icb != UInt64(0)
        Metal.concurrent_dispatch_threadgroups!(
            reinterpret(Core.LLVMPtr{Nothing,Metal.AS.Device}, icb),
            unsafe_load(p, 7), (groups, UInt32(1), UInt32(1)),
            (unsafe_load(p, 8), UInt32(1), UInt32(1)))
    end
    return nothing
end

"""
Retire a plan's outgoing argument memory (after a refit) without freeing it under
a frame in flight. It is outside the pool, so it becomes a root of `dev`'s open
batch and is released when that command buffer completes.
"""
function Mantle.retire!(::Pool, dev::MetalDevice, buf::MTL.MTLBuffer)
    # `dev`'s batch, not `Metal.global_queue` or the process device: the root must
    # be held by the command buffer that reads these bytes.
    Metal.record_operation!(batchqueue(dev), buf)
    return nothing
end

"""
One argument of a recorded dispatch: binding slot `index` and `nbytes` at `offset`
in the plan's argument memory, fixed at compile. A non-`nothing` `buffer` (an
`MTLBuffer` or `MtlPtr` argument) binds itself at `bufoffset` instead.
"""
struct RecordedArg
    index::Int
    offset::Int                       # into the plan's argument memory
    nbytes::Int
    buffer::Union{Nothing,MTL.MTLBuffer}
    bufoffset::Int
end

"""
A dispatch compiled for recording: pipeline, argument bindings, grid, and the
`adapted` values `pack_recorded!` writes at `record!`, held to avoid a per-launch
`mtlconvert`. Core lays out the argument memory from `argsize`.
"""
struct MetalRecordedDispatch{A<:Tuple,R<:Tuple,G,T,D,X<:Tuple,C<:Tuple}
    kernel::Metal.HostKernel
    args::Vector{RecordedArg}
    adapted::A
    # The arguments before `mtlconvert` at compile, whose Metal.jl records a
    # recording claims (`argmanaged`).
    raw::R
    # The device and the declared arguments, resolved again at every walked run:
    # a resource may have moved since compile. `kactx` is a KA kernel's context
    # argument, empty for a KernelInterface kernel.
    dev::D
    declared::X
    kactx::C
    state::Metal.KernelState
    groups::G
    nthreads::T
    name::String
    argoff::Int
    argsize::Int
    # Core's indirect slot for a `DeviceRange` dispatch in a recorded plan, whose
    # grid the device writes each run (`writegroups!`); zero otherwise.
    indirect::Int
    # The declared ndrange; core's prepare reads the count resource from it.
    ndrange::Any
end

"""
Run this dispatch now, as a walked plan (one that draws) does. Passes the resolved
arguments unconverted: `encode_arguments!` converting an `MtlArray` calls
`useResource`, which feeds hazard tracking; pre-converted values skip it, and the
driver may overlap the next command buffer (a render pass then reads a stale
indirect count).
"""
function (d::MetalRecordedDispatch)()
    d.kernel(d.kactx..., Mantle.resolve(d.dev, d.declared)...; groups = d.groups, threads = d.nthreads)
    return nothing
end

Mantle.argsize(d::MetalRecordedDispatch) = d.argsize
# A `DeviceRange` dispatch in a recorded plan is sized on the device; see
# `MetalIndirectSlot`.
Mantle.indirectindex(d::MetalRecordedDispatch) = d.indirect
Mantle.devicesized(d::MetalRecordedDispatch) = d.indirect != 0
Mantle.workgroupsize(d::MetalRecordedDispatch) = UInt32(threadsx(d.nthreads))

"""Threads per group along x, which a 1-D device count divides by."""
threadsx(n::Integer) = n
threadsx(t::Tuple) = first(t)
threadsx(s::MTL.MTLSize) = s.width

"""
An entry-point name Metal accepts: `mantle_` plus `name` with non-identifier
characters replaced (`repeat!/gate-1` does not link). A recorded dispatch is named
by its kernel alone because the name is part of the compile key; the slotted form
is for the recorder's own helpers.
"""
icb_name(name::AbstractString) = "mantle_" * identifier(name)
icb_name(name::AbstractString, i::Int) = icb_name(name) * "_" * string(i)

# The argument types whose slot binds the allocation itself, not a copy of bytes.
Mantle.passedas(::Type{<:MTL.MTLBuffer}) = Mantle.DeviceAllocation()
Mantle.passedas(::Type{<:Metal.MtlPtr}) = Mantle.DeviceAllocation()

"""The buffer and byte offset a `DeviceAllocation` argument binds; covers the two
types `passedas` names."""
boundbuffer(b::MTL.MTLBuffer) = (b, 0)
boundbuffer(p::Metal.MtlPtr) = (p.buffer, Int(p.offset))

"""
    recorded_args(adapted, argoff) -> (args, nbytes)

Each argument's binding slot and byte offset from `argoff`, done once at compile.
Slots come from [`Mantle.eachpassedarg`](@ref); offsets are `argalign`ed so any can
back an indirect command. Buffer arguments are made persistently resident.
"""
function recorded_args(adapted::Tuple, argoff::Int)
    out = RecordedArg[]
    off = argoff
    Mantle.eachpassedarg(adapted) do slot, a, passing
        if passing isa Mantle.DeviceAllocation
            buf, bufoff = boundbuffer(a)
            checkresident(buf)
            Metal.make_persistently_resident!(buf)
            push!(out, RecordedArg(slot, 0, 0, buf, bufoff))
        else
            n = sizeof(typeof(a))
            push!(out, RecordedArg(slot, off, n, nothing, 0))
            off += Mantle.argalign(n)
        end
        return nothing
    end
    return out, off - argoff
end

"""
Throw for a buffer under 8 bytes: an indirect command reaching it reads zeros and
drops writes silently, though an ordinary dispatch works.
"""
function checkresident(buf::MTL.MTLBuffer)
    buf.length >= 8 && return nothing
    throw(ArgumentError(
        "record!: a $(buf.length)-byte buffer is bound into a recorded command. " *
        "A resource an indirect command reaches by address is not made resident " *
        "below eight bytes: it reads as zero and its writes are dropped. Allocate " *
        "it through the device's pool, whose blocks are megabytes."))
end

"""
Write one dispatch's argument bytes into the plan's memory. With `pl`/`target`
(a recorded dispatch) each packed argument goes to core's `notepacked!`, so a
`resize!` can patch device pointers in place; writers pass `nothing`, since they
address the recording's own buffers, which never move.
"""
function pack_recorded!(ptr::Ptr{UInt8}, args::Vector{RecordedArg}, adapted::Tuple,
                        pl = nothing, target = nothing)
    # The same walk as `recorded_args`, so `args[slot]` is this argument's entry.
    Mantle.eachpassedarg(adapted) do slot, a, passing
        passing isa Mantle.DeviceAllocation && return nothing   # binds itself
        rec = args[slot]
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ptr + rec.offset, 0, rec.nbytes)
        Mantle.storebits!(ptr + rec.offset, a)
        pl === nothing || Mantle.notepacked!(pl, typeof(a), ptr + rec.offset, target, rec.offset)
        return nothing
    end
    return nothing
end

"""
The kernel state a recorded dispatch carries, built once (a launch builds one per
launch). Its seed is chosen once, so Metal's device RNG repeats its stream every frame.
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
The ndrange a recorded dispatch is compiled for: a `DeviceRange` becomes its
ceiling (core's `dispatchrange`). A recorded plan narrows it to the count each run
(`writegroups!`); a walked plan relies on the kernel's required bounds check.
"""
recordedrange(nd) = Mantle.dispatchrange(nd)

"""
    Mantle.compile_dispatch(compile, dispatch, argoff, indirect)

One dispatch compiled into a command the recorder can encode: iteration space,
grid, argument conversion, kernel state and pipeline, all worked out once.
"""
function Mantle.compile_dispatch(c::Mantle.Compile{<:MetalDevice}, d::Mantle.Dispatch,
                                 argoff::Int, indirect::Int)
    dev = c.graph.dev
    # A walked plan has no recording to rewrite: it dispatches the `DeviceRange`
    # ceiling, and without a ceiling core's interpreted launch reads the count on
    # the host.
    walked = !Mantle.recordablepasses(c.graph.dev, c.graph.passes)
    d.ndrange isa Mantle.DeviceRange && walked && d.ndrange.max === nothing &&
        return Mantle.bake(c, d)
    indirect = walked ? 0 : indirect
    # A declared call is not a `@kernel`: core's `bake` resolves it into a `Call`
    # (see the three-argument `dispatch!`). `kernelfor` would throw a `MethodError`.
    Mantle.iscall(d) && return Mantle.bake(c, d)
    # A macro-free KernelInterface kernel has no KA context; compiled here so its
    # pipeline gets `indirect = true`, which the KI wrapper does not set.
    if !Mantle.buildskernel(d.kernel, Mantle.backend(dev))
        raw = map(a -> Mantle.resolve(dev, a), d.args)
        adapted = map(Metal.mtlconvert, raw)
        tt = Tuple{map(Core.Typeof, adapted)...}
        entry = icb_name(string(nameof(typeof(d.kernel))))
        kernel = Metal.mtlfunction(Metal.mtlconvert(d.kernel), tt;
                                   name = entry, indirect = true)
        wrapped = KI.Kernel(Mantle.backend(dev), kernel)
        nd = recordedrange(d.ndrange)
        threads, groups = Mantle.ki_launch_extents(wrapped, nd, d.group)
        kernel.loggingEnabled && throw(ArgumentError(
            "record!: kernel $(entry) uses device-side logging, which needs a per-launch buffer"))
        state = recorded_state(dev, kernel)
        recargs, nbytes = recorded_args((state, d.kernel, adapted...), argoff)
        return MetalRecordedDispatch(kernel, recargs, adapted, raw, dev, d.args, (), state,
                                     groups, threads, entry, argoff, nbytes, indirect,
                                     d.ndrange)
    end
    obj = Mantle.kernelfor(d, dev)
    raw = map(a -> Mantle.resolve(dev, a), d.args)
    args = map(Metal.mtlconvert, raw)
    nd = recordedrange(d.ndrange)
    # `callgroup`, not `nothing`: a `DynamicSize` kernel's launch workgroup size
    # lives only there, and a different size here would desynchronise a body that
    # reads its size from a `Val` argument.
    ndrange, workgroupsize, iterspace, _ = KA.launch_config(obj, nd,
                                                            Mantle.callgroup(obj, d.group))
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    entry = icb_name(string(nameof(typeof(obj.f))))
    adapted = (Metal.mtlconvert(ctx), args...)
    tt = Tuple{map(Core.Typeof, adapted)...}
    kernel = Metal.mtlfunction(obj.f, tt; name = entry, indirect = true)
    # A dynamic workgroup size gets core's `defaultgroup`, capped by the pipeline's
    # `maxthreads`, and the context is rebuilt for that partition.
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        ws = Mantle.defaultgroup(ndrange, kernel.maxthreads)
        iterspace, _ = KA.partition(obj, ndrange, ws)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
        adapted = (Metal.mtlconvert(ctx), args...)
        tt2 = Tuple{map(Core.Typeof, adapted)...}
        # Recompile if the context type changed: a pipeline encoded with another
        # context type's bytes reads its ndrange from the wrong fields.
        tt2 === tt || (kernel = Metal.mtlfunction(obj.f, tt2; name = entry, indirect = true))
    end
    kernel.loggingEnabled && throw(ArgumentError(
        "record!: kernel $(entry) was compiled with device-side logging, whose " *
        "launch path allocates a log buffer per launch. A recorded plan has no " *
        "per-launch path; run the plan without recording it to use the log."))
    state = recorded_state(dev, kernel)
    recargs, nbytes = recorded_args((state, obj.f, adapted...), argoff)
    return MetalRecordedDispatch(kernel, recargs, adapted, raw, dev, d.args, (adapted[1],), state,
                                 length(KA.blocks(iterspace)),
                                 length(KA.workitems(iterspace)),
                                 entry, argoff, nbytes, indirect, d.ndrange)
end

"""
A run of commands a frame replays with one call. `slot` is the range buffer's
`(location, length)` pair gating it, or `-1` for a segment that always runs; the
gate decides how many of its commands the command processor runs.
"""
struct MetalSegment
    first::Int          # one-based, into the indirect command buffer
    last::Int
    slot::Int           # zero-based pair index into the range buffer, or -1
end

"""
The one-thread command that writes a gated segment's run. Its last two arguments
are patched when the segment closes (`patchwriter!`): the writer is encoded before
the commands it counts. The kernel `F` is `metal_write_range!` for a replayed recording, or
`metal_write_grids!` for a queue that encodes one: a gated dispatch then reads its
threadgroup count from memory (`dispatchThreadgroupsWithIndirectBuffer:`), and a
closed gate writes zero.
"""
mutable struct MetalWriter{F}
    kernel::Metal.HostKernel
    args::Vector{RecordedArg}
    adapted::Tuple
    state::Metal.KernelState
end

"""A recorded dispatch's buffer bindings for a legacy encoder: `(buffer, offset, index)`."""
const LegacyBindings = Vector{Tuple{MTL.MTLBuffer,Int,Int}}

"""
One recorded dispatch as an encoder needs it, for a queue that encodes instead of
replaying (`encodes` in `device.jl`). Fixed at record: pipeline, grid and the
argument bindings, so a frame is three sends per dispatch.
"""
struct MetalEncodedCommand
    pipeline::MTL.MTLComputePipelineState
    table::Union{MTL.MTL4ArgumentTable,LegacyBindings}
    groups::MTL.MTLSize
    threads::MTL.MTLSize
    barrier::Bool
    # Zero-based word offset into the grid buffer for a dispatch in a gated
    # segment (device-written count); `-1` for a grid fixed at record.
    gridoff::Int
    # Byte offset of a device-sized dispatch's slot in the argument memory; `-1`
    # for every other command.
    slotoff::Int
end

"""The same command with its barrier bit set."""
withbarrier(c::MetalEncodedCommand) =
    MetalEncodedCommand(c.pipeline, c.table, c.groups, c.threads, true, c.gridoff,
                        c.slotoff)

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
    # Filled only when the queue encodes; empty means no encoded form.
    encoded::Vector{MetalEncodedCommand}
    segments::Vector{MetalSegment}
    # The plan's host calls, each paired with how many segments closed before it
    # (`0` runs before the first segment).
    calls::Vector{Pair{Int,Any}}
    # The segment being built: first command, predicate, range slot.
    first::Int
    pred::Any
    slot::Int
    # The writers' own arguments, which the plan's argument memory does not cover.
    aux::MTL.MTLBuffer
    auxptr::Ptr{UInt8}
    auxcursor::Int
    ranges::Metal.MtlVector{UInt32}
    # Encode path: three `UInt32` threadgroup counts per gated dispatch. `templ`
    # holds the recorded grids an open gate restores, filled from `templhost` when
    # the recording closes. One element each on a queue that replays.
    grids::Metal.MtlVector{UInt32}
    templ::Metal.MtlVector{UInt32}
    templhost::Vector{UInt32}
    gridcursor::Int     # triples allocated so far
    gridfirst::Int      # the first triple of the segment under construction
    writers::Vector{MetalWriter}
    nwriters::Int       # how many gated runs the plan holds, counted at open
end

"""What a recorded frame replays, and everything it keeps alive for it."""
struct MetalRecording
    icb::MTL.MTLIndirectCommandBuffer
    segments::Vector{MetalSegment}
    # `segments closed before => call`, in plan order (`replaywithcalls!`).
    calls::Vector{Pair{Int,Any}}
    rangebuf::MTL.MTLBuffer
    rangeoff::Int
    ranges::Metal.MtlVector{UInt32}
    # Grid buffer and template for gated encoded dispatches (`MetalGridWriter`).
    gridbuf::MTL.MTLBuffer
    gridoff::Int
    grids::Metal.MtlVector{UInt32}
    templ::Metal.MtlVector{UInt32}
    aux::MTL.MTLBuffer
    argstore::MTL.MTLBuffer
    writers::Vector{MetalWriter}
    ncommands::Int
    # Empty unless the queue encodes (`MetalEncodedCommand`).
    encoded::Vector{MetalEncodedCommand}
    # What the replay declares to its encoder (`ensureresident!`).
    resources::Vector{MTL.MTLBuffer}
    # `resources` as the object pointers `useResources:` takes, so a frame does
    # not convert it. Rebuilt only together; `resources` keeps the objects alive.
    residentids::Vector{MTL.ObjectiveC.id{MTL.MTLBuffer}}
    # The pool blocks' use records (`managedbuffer`), rebuilt with `resources` and
    # claimed by each replay so host reads wait for it (`claimbuffers!`).
    managed::Vector{Metal.Managed}
    nblocks::Base.RefValue{Int}
    # The same for Metal.jl-allocated arrays the commands reach, collected when
    # the recording closes (`argmanaged`).
    argmanaged::Vector{Metal.Managed}
end

"""
Collects, through `holdleaves!`, the use record of each Metal.jl array a plan's
arguments reach, once each.
"""
struct ManagedLeaves
    records::Vector{Metal.Managed}
end
function Mantle.hold!(c::ManagedLeaves, a::Metal.MtlArray)
    m = a.data[]
    any(r -> r === m, c.records) || push!(c.records, m)
    return a
end
# A pool region's block is claimed through `managedbuffer` already.
Mantle.hold!(::ManagedLeaves, a) = a

"""
    argmanaged(plan) -> Vector{Metal.Managed}

The use records of Metal.jl-allocated arrays the plan's recorded commands reach.
Commands name arrays by address, so Metal.jl never sees them used; a replay claims
these records so a host access through an `MtlArray` waits for it.
"""
function argmanaged(pl::Mantle.Plan)
    c = ManagedLeaves(Metal.Managed[])
    for pp in pl.passes, d in pp.dispatches
        d isa MetalRecordedDispatch && Mantle.holdleaves!(c, d.raw)
    end
    return c.records
end

# The recording's Objective-C objects die with it; the argument memory is the plan's.
Mantle.release!(::MetalRecording) = nothing

"""Why a plan this backend would otherwise record cannot be, or `nothing`."""
function norecordreason(pl::Mantle.Plan)
    for pp in pl.passes
        for d in pp.dispatches
            d isa MetalRecordedDispatch || d isa Mantle.Call ||
                return "pass \"$(pp.pass.name)\" holds a $(typeof(d)), which " *
                "this backend compiled for an interpreted run and cannot record."
        end
    end
    return nothing
end

"""
Sizes for recording `passes`: dispatches, range writers, calls, prepares and the
prepares' argument bytes. Per range because core may partition a plan
(`partitionranges`) and each piece is sized for itself. A call takes no command
and no argument memory, only a segment boundary.
"""
function planshape(pl::Mantle.Plan, passes::AbstractUnitRange = eachindex(pl.passes))
    ndispatch = 0
    nwriters = 0
    ncalls = 0
    nprepares = 0
    prepbytes = 0
    prev = nothing
    for i in passes
        pp = pl.passes[i]
        pp.pass.kind === :update && continue
        for d in pp.dispatches
            d isa Mantle.Call ? (ncalls += 1) : (ndispatch += 1)
        end
        # Core's prepare gets a segment of its own (`emitkernel!`,
        # `emitpreparebarrier!`), so this pass's gate needs a new writer.
        if pp.indirect
            nprepares += 1
            nind = Base.count(Mantle.devicesized, pp.dispatches)
            # Kernel state and gate, plus three per-dispatch tuples of up to 48
            # bytes an entry, each argument 256-byte aligned.
            prepbytes += 2 * 256 + 3 * Mantle.argalign(48 * nind)
            prev = nothing
        end
        pred = pp.pass.predicate
        pred === nothing || samepredicate(pred, prev) || (nwriters += 1)
        prev = pred
    end
    return (; ndispatch, nwriters, ncalls, nprepares, prepbytes)
end

"""
Whether two passes share a gate, and so a segment: same flag object and index.
"""
samepredicate(a::Tuple{Any,Int}, b::Tuple{Any,Int}) = a[1] === b[1] && a[2] == b[2]
samepredicate(::Tuple{Any,Int}, ::Nothing) = false
samepredicate(::Nothing, ::Any) = false

"""
    Mantle.openrecording(dev, plan, passes = eachindex(plan.passes)) -> MetalRecorder or nothing

A recorder with an indirect command buffer sized for `passes`, or `nothing` for a
walked plan. Throws for a plan that cannot be recorded.
"""
function Mantle.openrecording(d::MetalDevice, pl::Mantle.Plan,
                              passes::AbstractUnitRange = eachindex(pl.passes))
    Mantle.recordable(d, pl) || return nothing
    # Adopt before touching the GPU: recording launches fills and blits the gate
    # template, which on another task's queue would be unordered against the
    # frames. `templ` is written only once, so a lost blit opens every gate onto
    # a zero grid and the plan silently runs nothing.
    adoptqueue!(d)
    let why = norecordreason(pl)
        why === nothing || throw(ArgumentError("record!: " * why))
    end
    Metal.can_use_residency_sets(d.dev) || throw(ArgumentError(
        "record!: this device has no residency sets, and every buffer a recorded " *
        "command reaches is reached by address — without one nothing it reads is " *
        "resident. Run the plan without recording it."))
    (; ndispatch, nwriters, ncalls, nprepares, prepbytes) = planshape(pl, passes)
    # Only the whole plan must have dispatches: a graph opening on a loop gets an
    # empty head piece for `emithead!`, recorded like any other.
    ndispatch + ncalls > 0 || passes != eachindex(pl.passes) ||
        throw(ArgumentError("record!: the plan has no dispatches to record."))
    nslots = maximum(pp -> maximum(d -> maxslot(d), pp.dispatches; init = 0),
                     pl.passes; init = 0)
    # The descriptor bounds every command's bindings, writers' and prepares'
    # included; a command binding past it silently does nothing.
    nwriters == 0 || (nslots = max(nslots, RANGE_WRITER_SLOTS))
    nprepares == 0 || (nslots = max(nslots, PREPARE_SLOTS))
    nslots <= MAX_KERNEL_BUFFERS || throw(ArgumentError(
        "record!: a dispatch binds $(nslots) buffers and an indirect command may " *
        "bind $(MAX_KERNEL_BUFFERS). Pass fewer arguments, or group them in a struct."))
    # Core emits the range reset (`emithead!`) into the first piece only, so only
    # it reserves a command. A later piece with writers would replay stale ranges
    # and is refused, except a `repeat!` body: each iteration is its own replay,
    # and every writer rewrites its range, zero length when its gate is closed.
    head = first(passes) == 1 && nwriters > 0
    inloop = !isempty(passes) && pl.passes[first(passes)].pass.loop !== nothing
    nwriters == 0 || first(passes) == 1 || inloop || throw(ArgumentError(
        "record!: passes $(passes) hold $(nwriters) gated pass(es) and this is not " *
        "the first piece of the partition. A piece owns the range array its gates " *
        "write, and only the first is given the reset that clears it — so a later " *
        "gated piece would replay stale counts. Keep the gated passes in its first " *
        "$(first(passes) - 1) passes."))
    ncommands = ndispatch + nwriters + nprepares + (head ? 1 : 0)
    # `ray_tracing` always: in a buffer that did not declare it every traced ray
    # misses silently, and whether a kernel traces is not visible from the dispatch.
    desc = MTL.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = max(nslots, 1),
                                                  ray_tracing = true)
    # Shared because the host encodes the commands: API validation
    # (`MTL_DEBUG_LAYER=1`) refuses CPU access to a `Private` one. `max(.., 1)`
    # since a calls-only plan has no commands; `encode!` still bounds by `ncommands`.
    icb = MTL.MTLIndirectCommandBuffer(d.dev, desc, max(ncommands, 1);
                                       storage = MTL.MTLResourceStorageModeShared)
    # Resident for the recording's life: on MTL4 nothing else makes it resident,
    # and reading an unmapped indirect command buffer faults the queue.
    Metal.make_persistently_resident!(icb, d.dev)
    # One 256-byte slot per argument: the range reset takes three, each range
    # writer seven, the prepares `prepbytes`. Sized generously, once.
    aux = MTL.MTLBuffer(d.dev, max(256 * (4 + 7 * nwriters) + prepbytes, 256);
                        storage = Metal.SharedStorage)
    Metal.make_persistently_resident!(aux)
    ranges = Metal.MtlVector{UInt32}(undef, max(2 * nwriters, 2))
    fill!(ranges, UInt32(0))
    # Three words per command covers every dispatch being gated. One element on a
    # replaying queue, where nothing reads them; kept so the type stays concrete.
    ngrid = encodes(d.queue) && nwriters > 0 ? 3 * ncommands : 1
    grids = Metal.MtlVector{UInt32}(undef, ngrid)
    templ = Metal.MtlVector{UInt32}(undef, ngrid)
    fill!(grids, UInt32(0))
    fill!(templ, UInt32(0))
    return MetalRecorder(d, pl, icb, ncommands, 0, false, MetalEncodedCommand[],
                         MetalSegment[], Pair{Int,Any}[], 1, nothing, -1,
                         aux, convert(Ptr{UInt8}, MTL.contents(aux)), 0,
                         ranges, grids, templ, zeros(UInt32, ngrid), 0, 0,
                         MetalWriter[], nwriters)
end

"""The highest binding slot a dispatch uses, which the descriptor must declare."""
maxslot(d::MetalRecordedDispatch) = isempty(d.args) ? 0 : maximum(a -> a.index, d.args)
# A call binds nothing; the library lays out its own arguments.
maxslot(::Mantle.Call) = 0

"""Metal's own limit on how many buffers one command may bind."""
const MAX_KERNEL_BUFFERS = 31

"""Buffer slots core's prepare binds: kernel state, slots, counts, workgroup sizes
and gate."""
const PREPARE_SLOTS = 6

"""
Buffer slots a range writer binds: kernel state plus the six arguments of
`metal_write_range!` (the function itself is a ghost type and takes none).
"""
const RANGE_WRITER_SLOTS = 7

"""
One thread zeroing every execution range at the head of the frame: a discarded
iteration does not run the writer of the next range, which must then already say
"run nothing" rather than whatever the previous frame left.
"""
function metal_reset_ranges!(range, n::Int32)
    @inbounds for k in Int32(1):n
        range[k] = UInt32(0)
    end
    return nothing
end

"""
One thread writing the execution range the command processor reads next: at
zero-based word `base`, `loc` (zero-based first command) and `len`, or length zero
when the gate `pred[i + 1]` is closed.
"""
function metal_write_range!(range, pred, i::Int32, base::Int32, loc::UInt32, len::UInt32)
    @inbounds go = pred[i + Int32(1)].go != UInt32(0)
    @inbounds range[base + Int32(1)] = loc
    @inbounds range[base + Int32(2)] = go ? len : UInt32(0)
    return nothing
end

"""
The encode path's `metal_write_range!`: for `n` dispatches from zero-based word
`base`, copy the recorded grids from `templ` when the gate is open, else write
zeros so they run no threadgroups.
"""
function metal_write_grids!(grids, templ, pred, i::Int32, base::Int32, n::Int32)
    @inbounds go = pred[i + Int32(1)].go != UInt32(0)
    for k in Int32(1):(Int32(3) * n)
        @inbounds grids[base + k] = go ? templ[base + k] : UInt32(0)
    end
    return nothing
end

"""
    Mantle.emitdispatch!(recorder, dispatch, name)

Encode one dispatch as the next command of the recording.
"""
function Mantle.emitdispatch!(e::MetalRecorder, d::MetalRecordedDispatch,
                              name::AbstractString)
    # Metal's `argbytes` always lays out argument memory.
    am = e.plan.args::Mantle.ArgMemory
    pack_recorded!(am.ptr, d.args, (d.state, d.kernel.f, d.adapted...), e.plan, am)
    slotoff = d.indirect == 0 ? -1 : writeslot!(e, am, d)
    encode!(e, d.kernel.pipeline, d.args, am.store,
            MTL.MTLSize(d.groups), MTL.MTLSize(d.nthreads); slotoff)
    return nothing
end

"""
Write into `d`'s slot what its prepare needs to rewrite the command `encode!` writes
next (zero-based index = cursor before it advances); return the slot's byte offset.
An encoding queue gets a null handle, having no command to rewrite.
"""
function writeslot!(e::MetalRecorder, am, d::MetalRecordedDispatch)
    s = am.indirect[d.indirect]::MetalIndirectSlot
    p = am.ptr + s.offset
    icb = encodes(e.dev.queue) ? UInt64(0) : reinterpret(UInt64, e.icb.gpuResourceID)
    unsafe_store!(Ptr{UInt64}(p + 16), icb)
    unsafe_store!(Ptr{UInt32}(p + 24), UInt32(e.cursor))
    unsafe_store!(Ptr{UInt32}(p + 28), UInt32(threadsx(d.nthreads)))
    return s.offset
end

"""
    Mantle.emitdispatch!(recorder, call, name)

Close the segment at a host call and remember it; `replaywithcalls!` runs the call
between the two `executeCommandsInBuffer:` calls, ordered by encoder order. Refused
inside a gated pass, since the host cannot skip a call for a discarded iteration.
"""
function Mantle.emitdispatch!(e::MetalRecorder, c::Mantle.Call,
                              name::AbstractString)
    e.pred === nothing || throw(ArgumentError(
        "record!: pass \"$name\" declares a library call inside a gated pass. " *
        "Whether a gated pass runs is decided on the device and the host cannot " *
        "read it, so a host call in one would run for an iteration that does not. " *
        "Declare the operation as dispatches inside the gate, or move the call out."))
    closesegment!(e)
    push!(e.calls, length(e.segments) => c)
    return nothing
end

"""
Encode the next command: pipeline, one buffer per argument slot (a slice of `store`,
or the argument's own buffer), grid, and the pending barrier. `e.barrier` is set by
`emitbarriers!` and consumed here, so only a pass's first command carries it.
"""
function encode!(e::MetalRecorder, pipeline::MTL.MTLComputePipelineState,
                 args::Vector{RecordedArg}, store::MTL.MTLBuffer,
                 groups::MTL.MTLSize, threads::MTL.MTLSize; slotoff::Int = -1)
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
    # Also captured for encoding, for a queue that does not replay; the addresses
    # are fixed, so the table is built once.
    if encodes(e.dev.queue)
        # A gated command reads its grid from a triple; a device-sized one from
        # its slot, which the prepare zeroes for a discarded iteration.
        goff = -1
        if e.slot >= 0 && slotoff < 0
            goff = 3 * e.gridcursor
            e.gridcursor += 1
            e.templhost[goff + 1] = UInt32(groups.width)
            e.templhost[goff + 2] = UInt32(groups.height)
            e.templhost[goff + 3] = UInt32(groups.depth)
        end
        push!(e.encoded,
              MetalEncodedCommand(pipeline, argtable(e.dev, args, store), groups,
                                  threads, e.barrier, goff, slotoff))
    end
    # MTL4 honours a command's barrier bit despite having no hazard tracking, so
    # an execute need not be split at barriers.
    if e.barrier
        MTL.set_barrier!(cmd)
        e.barrier = false
    end
    return cmd
end

"""
    Mantle.emitkernel!(recorder, f, args...; ndrange, workgroup_size)

Record core's prepare kernel for a pass's device-sized dispatches as the next
command, arguments in the auxiliary memory. Into an ungated segment, so it also
runs for a discarded iteration and writes zero groups.
"""
function Mantle.emitkernel!(e::MetalRecorder, f, args...; ndrange, workgroup_size)
    e.pred === nothing || closesegment!(e)
    threads = prod(workgroup_size)
    auxcommand!(e, f, map(Metal.mtlconvert, args), icb_name(string(nameof(f)));
                groups = cld(ndrange, threads), threads)
    return nothing
end

"""
    auxcommand!(e, f, adapted, name; groups = 1, threads = 1, pack = true) -> (kernel, args, state)

Compile `f` for `adapted`, lay its arguments out in the recording's auxiliary
memory and encode it as the next command. `pack = false` leaves the bytes to a
later `patchwriter!`.
"""
function auxcommand!(e::MetalRecorder, f, adapted::Tuple, name::String;
                     groups::Integer = 1, threads::Integer = 1, pack::Bool = true)
    kernel = Metal.mtlfunction(f, Tuple{map(Core.Typeof, adapted)...}; name, indirect = true)
    state = recorded_state(e.dev, kernel)
    args, nbytes = recorded_args((state, f, adapted...), e.auxcursor)
    e.auxcursor += Mantle.argalign(nbytes)
    e.auxcursor <= Int(e.aux.length) || error(
        "record!: the recording's auxiliary memory holds $(Int(e.aux.length)) bytes and " *
        "its writers and prepares need $(e.auxcursor); `planshape` sized it short.")
    pack && pack_recorded!(e.auxptr, args, (state, f, adapted...))
    encode!(e, kernel.pipeline, args, e.aux, MTL.MTLSize(groups), MTL.MTLSize(threads))
    return kernel, args, state
end

"""
After the prepare: close the segment, since rewritten commands are read from the
next `execute`, and set a barrier for the prepare's writes.
"""
function Mantle.emitpreparebarrier!(e::MetalRecorder)
    closesegment!(e)
    e.barrier = true
    return nothing
end

"""
    Mantle.passbarriers(compile, pass) -> (images, transitions, barrier)

The hazards the `Barriers` phase derived for this pass, kept because commands in
an indirect command buffer have no commit order. No image transitions and no
barrier object: the recorder only sets a command's barrier bit from the list.
"""
Mantle.passbarriers(c::Mantle.Compile{<:MetalDevice}, p::Mantle.Pass) =
    (Nothing[], c.prepass[p], nothing)

"""
Set the barrier bit for the pass's first command when the graph derived a hazard.
A command with the bit waits for every earlier command in the replay; the rest of
the pass runs concurrently.
"""
function Mantle.emitbarriers!(e::MetalRecorder, pp::Mantle.PassPlan)
    isempty(pp.pre) || (e.barrier = true)
    return nothing
end

# A recording holds no host store (it would replay this run's values forever);
# core lands them in each run's submission (`emitupdates!`).
Mantle.emitupdate!(::MetalRecorder, ::Mantle.Plan, ::Mantle.PassPlan) = nothing

# Apple GPUs sample counters only at stage boundaries and a replay is one command
# buffer for all passes, so a recorded pass has no GPU time of its own.
Mantle.passend!(::Mantle.Plan, ::MetalRecorder, ::Integer, _) = NaN

"""
Open a segment for this pass's predicate unless it continues the current one. Any
change of gate, to or from none, ends a segment. The range writer goes into an
ungated segment before the gated one, so it runs on every replay before its range
is read.
"""
function Mantle.withpredicate(f, e::MetalRecorder, pred::Tuple{Any,Int})
    if !samepredicate(pred, e.pred)
        # Never into another gated segment: a closed gate there would discard the
        # writer and leave this range unwritten.
        e.pred === nothing || closesegment!(e)
        # The writer waits for the open segment, which holds the gate dispatch.
        e.barrier = true
        emitwriter!(e, pred)
        closesegment!(e)
        e.pred = pred
        e.slot = length(e.writers) - 1
        # Where this segment's triples begin, read by `patchwriter!` at close.
        e.gridfirst = e.gridcursor
    end
    f()
    return nothing
end

function Mantle.withpredicate(f, e::MetalRecorder, ::Nothing)
    e.pred === nothing || closesegment!(e)
    f()
    return nothing
end

"""Close the segment under construction, if it holds anything."""
function closesegment!(e::MetalRecorder)
    if e.cursor >= e.first
        push!(e.segments, MetalSegment(e.first, e.cursor, e.slot))
        # The writer was encoded before the commands it counts; patch it now.
        e.slot < 0 || patchwriter!(e, e.writers[e.slot + 1], e.first, e.cursor)
    end
    e.first = e.cursor + 1
    e.pred = nothing
    e.slot = -1
    return nothing
end

"""
Encode the command that decides on the device whether a gated segment runs: an
execution range writer for a replay, a threadgroup-count writer when the queue
`encodes`.
"""
# The last two arguments, `base` and `n`, are patched when the segment closes; zero
# until then, which writes nothing.
function emitwriter!(e::MetalRecorder, pred::Tuple{Any,Int})
    slot = length(e.writers)
    cond = (Metal.mtlconvert(Mantle.storage(pred[1])), Int32(pred[2]))
    f, adapted, name = encodes(e.dev.queue) ?
        (metal_write_grids!, (Metal.mtlconvert(e.grids), Metal.mtlconvert(e.templ), cond...,
                              Int32(0), Int32(0)), icb_name("write_grids", slot)) :
        (metal_write_range!, (Metal.mtlconvert(e.ranges), cond..., Int32(2 * slot),
                              UInt32(0), UInt32(0)), icb_name("write_range", slot))
    kernel, args, state = auxcommand!(e, f, adapted, name; pack = false)
    w = MetalWriter{typeof(f)}(kernel, args, adapted, state)
    push!(e.writers, w)
    return w
end

# The run a writer writes, as its last two arguments.
writertail(::MetalWriter{typeof(metal_write_range!)}, e, first, last) = (UInt32(first - 1), UInt32(last - first + 1))
writertail(::MetalWriter{typeof(metal_write_grids!)}, e, first, last) = (Int32(3 * e.gridfirst), Int32(last - first + 1))

"""Give a writer the run it writes, and pack its arguments."""
function patchwriter!(e::MetalRecorder, w::MetalWriter{F}, first::Int, last::Int) where {F}
    w.adapted = (w.adapted[1:end-2]..., writertail(w, e, first, last)...)
    pack_recorded!(e.auxptr, w.args, (w.state, F.instance, w.adapted...))
    return nothing
end

"""
    Mantle.emithead!(recorder, plan)

The range reset as the first command, for a plan with gates, so every `run!`
clears the ranges (or grids) before the first writer runs.
"""
function Mantle.emithead!(e::MetalRecorder, pl::Mantle.Plan)
    e.nwriters == 0 && return nothing
    # Same kernel for ranges and grids.
    arr = encodes(e.dev.queue) ? e.grids : e.ranges
    n = encodes(e.dev.queue) ? Int32(length(e.grids)) : Int32(2 * e.nwriters)
    auxcommand!(e, metal_reset_ranges!, (Metal.mtlconvert(arr), n), icb_name("reset_ranges", 0))
    return nothing
end

"""
    Mantle.emitloophead!(recorder, plan, body)

Before each iteration of a `repeat!` body: set the barrier bit. Each iteration is
its own replay, and replays are already ordered (MTL4 waits on the previous event;
legacy runs in order behind the resources `opensubmit!` declares).
"""
function Mantle.emitloophead!(e::MetalRecorder, ::Mantle.Plan, ::AbstractUnitRange)
    e.barrier = true
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
    am = pl.args::Mantle.ArgMemory
    if encodes(e.dev.queue)
        # The grids an open gate restores, uploaded once on the adopted queue.
        e.gridcursor == 0 || copyto!(e.templ, e.templhost)
        markgates!(e)
    end
    return MetalRecording(e.icb, e.segments, e.calls,
                          e.ranges.data[].buffer, Int(e.ranges.offset),
                          e.ranges, e.grids.data[].buffer, Int(e.grids.offset), e.grids,
                          e.templ, e.aux, am.store, e.writers, e.ncommands,
                          e.encoded, MTL.MTLBuffer[],
                          MTL.ObjectiveC.id{MTL.MTLBuffer}[], Metal.Managed[], Ref(-1),
                          argmanaged(pl))
end

"""
Set the barrier bit on the first command of every gated segment and on the command
after it, for the encoded frame: the counts a writer stores are read by the command
processor, which no in-shader ordering covers.
"""
function markgates!(e::MetalRecorder)
    for sg in e.segments
        sg.slot < 0 && continue
        for k in (sg.first, sg.last + 1)
            1 <= k <= length(e.encoded) && (e.encoded[k] = withbarrier(e.encoded[k]))
        end
    end
    return nothing
end

# Unreachable: plans with copy or render passes are walked, not recorded.
Mantle.emitcopy!(::MetalRecorder, ::Mantle.Plan, pp::Mantle.PassPlan) =
    error("record!: pass \"$(pp.pass.name)\" is a copy pass and reached the recorder.")
Mantle.beginrender!(::MetalRecorder, ::Mantle.Plan, pp::Mantle.PassPlan) =
    error("record!: pass \"$(pp.pass.name)\" is a render pass and reached the recorder.")

# `run!` refuses a recordable plan that was never `record!`ed; a plan with a render
# or copy pass is not recordable and is walked.
Mantle.recordsplans(::MetalDevice) = true
# Legacy declares every replay resource ReadWrite on its tracked encoder;
# MTL4 submissions wait on the previous event and the front's legacy stores.
# Both routes order loop entries via emitloophead!. Core can therefore omit
# redundant buffer-entry transitions without changing scheduling policy.
Mantle.buffer_head_ordered(::MetalDevice) = true
# An ICB records compute commands, not render/copy encoders. Core chooses the
# walked route from these native command capabilities before compiling passes.
Mantle.recordablekind(::MetalDevice, ::Val) = false
Mantle.recordablekind(::MetalDevice, ::Val{:compute}) = true
Mantle.recordablekind(::MetalDevice, ::Val{:update}) = true

"""
    Mantle.openrun(dev, plan) -> Immediate

The emitter for host stores or patches in front of a run's recording. Core opens
it only for such a run, and a recorded plan's run first waits for the device: the
stores land through the host pointer, unordered against frames still reading.
"""
function Mantle.openrun(d::MetalDevice, pl::Mantle.Plan)
    pl.recording === nothing || waitidle(d)
    enterrun!(d)
    return Mantle.Immediate()
end

"""
The head of every run: `adoptqueue!` for the running task and `ownflush!`. Called
from `submitrun!` too, since a steady recorded frame never reaches `openrun`.
Idempotent and cheap when the queue is already adopted.
"""
function enterrun!(d::MetalDevice)
    adoptqueue!(d)
    ownflush!(d, true)
    return nothing
end

"""
    Mantle.closerun!(dev, plan, emitter)

Submit the run behind whatever the front emitted, then present.
"""
Mantle.closerun!(d::MetalDevice, pl::Mantle.Plan, ::Mantle.Immediate) = submitrun!(d, pl)
# A run without a front gets `nothing`. Two methods, since a `Union` method would be
# ambiguous with core's `closerun!(dev, pl, ::Immediate)`.
Mantle.closerun!(d::MetalDevice, pl::Mantle.Plan, ::Nothing) = submitrun!(d, pl)

function submitrun!(d::MetalDevice, pl::Mantle.Plan)
    enterrun!(d)
    # The token `waitfor!(plan)` waits on comes from the submission. Through core's
    # `submitplan!`, which also handles a partitioned plan (`RecordingParts`).
    tok = Mantle.submitplan!(d, pl, nothing, MetalRecording)
    for s in pl.graph.surfaces
        Mantle.present_frame!(d, s.win)
    end
    # Restore Metal.jl's batching for launches outside a run. A run that threw
    # leaves it held, which is harmless: `synchronize` flushes regardless.
    ownflush!(d, false)
    return tok
end

"""
The resources a replay declares to its encoder, as object pointers. Commands reach
buffers by address, and an undeclared buffer gets no hazard tracking, so the driver
may overlap consecutive replays. Declares the pool's blocks (few, and covering every
graph resource), the Metal.jl arrays the plan reaches (`argmanaged`) and the
recording's own buffers. Rebuilt only when the pool's block set changed.
"""
function ensureresident!(d::MetalDevice, rec::MetalRecording)
    p = pool(d)
    # One comparison without the lock: `blockgen` is bumped wherever a block is
    # created or destroyed.
    p.blockgen[] == rec.nblocks[] && return rec.residentids
    # The generation is read under the lock with the copy, so a block added
    # meanwhile is not stamped as covered.
    n = lock(p.lock) do
        empty!(rec.resources)
        empty!(rec.managed)
        for (_, blocks) in p.blocks, b in blocks
            b.memory isa MTL.MTLBuffer || continue
            push!(rec.resources, b.memory)
            push!(rec.managed, managedbuffer(d, b.memory))
        end
        # Metal.jl's own arrays the commands reach, declared like the blocks.
        for m in rec.argmanaged
            push!(rec.resources, m.buffer)
            push!(rec.managed, m)
        end
        p.blockgen[]
    end
    # The plan's and the recording's own buffers, outside the pool.
    push!(rec.resources, rec.argstore)
    push!(rec.resources, rec.aux)
    push!(rec.resources, rec.rangebuf)
    push!(rec.resources, rec.gridbuf)
    push!(rec.resources, rec.templ.data[].buffer)
    # Residency is granted here, on rebuild only: granting per submission left the
    # GPU idle most of a small frame.
    foreach(Metal.make_persistently_resident!, rec.resources)
    resize!(rec.residentids, length(rec.resources))
    @inbounds for i in eachindex(rec.resources)
        rec.residentids[i] = Base.pointer(rec.resources[i])
    end
    rec.nblocks[] = n
    return rec.residentids
end

"""
Replay one recorded piece. The emitter argument is unused: host stores ride in the
run's own submission (`emitupdates!`).
"""
Mantle.submitrecording!(d::MetalDevice, rec::MetalRecording, ::Any) = replay!(d, rec)

"""
Replay the pieces core groups into one submission (`submissionbatches`) in one
command buffer, avoiding a command buffer and encoder allocation per piece.
"""
Mantle.submitrecording!(d::MetalDevice, recs::Vector{MetalRecording}, ::Any) = replay!(d, recs)

# A walked plan: its passes went into the batch, and closing the frame submits it.
Mantle.submitrecording!(d::MetalDevice, ::Nothing, ::Any) = closeframe!(d)

"""
One frame of a recorded plan: an encoder, one `execute` per segment of each piece
(or the encoded dispatches), and a commit. Returns the submission's token.
"""
replay!(d::MetalDevice, rec::MetalRecording) = replay!(d, (rec,))

function replay!(d::MetalDevice, recs::Union{Tuple{MetalRecording},Vector{MetalRecording}})
    # A recording shaped for the other route is refused (`canrun`). Residency for
    # every piece comes before the submission names the set, which does not pick
    # up later allocations.
    for rec in recs
        canrun(d.queue, rec) || refusereplay(d.queue)
        ensureresident!(d, rec)
    end
    for rec in recs
        claimbuffers!(batchqueue(d), rec.managed)
    end
    # Hold Metal.jl's `submission_lock` from `opensubmit!` to the commit: the batch
    # is shared by every task, and another thread's launch or flush landing in an
    # open encoder makes Metal abort the process. The in-flight wait comes after,
    # without the lock, as in Metal.jl's `flush!`.
    tok = Base.@lock Metal.submission_lock begin
        sub = opensubmit!(d, first(recs).residentids)
        for k in eachindex(recs)
            rec = recs[k]
            k == firstindex(recs) || nextpiece!(d, sub, rec.residentids)
            sub = encodepiece!(d, sub, rec)
        end
        closesubmit!(d, sub)
    end
    Metal.limit_inflight!(batchqueue(d))
    return tok
end

"""Encode one piece into the open submission and return the submission, which a
piece holding calls has reopened."""
function encodepiece!(d::MetalDevice, sub, rec::MetalRecording)
    if !canreplay(d.queue)
        encodeplan!(sub, rec)
    elseif isempty(rec.calls)
        for s in rec.segments
            executesegment!(d, sub, rec, s)
        end
    else
        sub = replaywithcalls!(d, sub, rec, rec.residentids)
    end
    return sub
end

"""
Replay a recording with host calls: its segments in order, each followed by the
calls registered after it. The encoder is suspended around the calls and reopened,
not committed, so a library encoding into the command buffer joins this
submission; reopening redoes the per-encoder `useResource` declaration.
"""
function replaywithcalls!(d::MetalDevice, sub, rec::MetalRecording, ids)
    next = 1
    ncalls = length(rec.calls)
    # `0` is before the first segment.
    for i in 0:length(rec.segments)
        i == 0 || executesegment!(d, sub, rec, rec.segments[i])
        next <= ncalls && first(rec.calls[next]) == i || continue
        suspendsubmit!(d, sub)
        while next <= ncalls && first(rec.calls[next]) == i
            last(rec.calls[next])()
            next += 1
        end
        sub = opensubmit!(d, ids)
    end
    next > ncalls || error(
        "replay!: $(ncalls - next + 1) of the recording's calls name a segment the " *
        "replay never reached; the recording and its segment list disagree.")
    return sub
end
