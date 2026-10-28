# Backend hooks for graph execution. Core owns scheduling, liveness, placement and
# barriers; backends compile, encode and submit the resulting work.

"""
    isdepth(x) -> Bool

Classify attachment element types: `Float32` is depth, every other type is color.
"""
isdepth(::Type{Float32}) = true
isdepth(::Type) = false
isdepth(x) = isdepth(typeof(x))

"""
    makeimage(device, T, width, height, srgb, usage, source) -> TransientImage

Create an unbound driver image and TransientImage with its memory requirements.
Do not allocate image memory. Initialize the empty liveness interval to (typemax(Int), 0).
"""
function makeimage end

"""
    remakeimage!(device, t)

Recreate the image on its owning device after an extent change and update t.req.
Retain the existing format and usage.
"""
function remakeimage! end

"""
    target_extent(target) -> (width, height)

Return attachment dimensions as a driver-independent tuple.
"""
function target_extent end

"""
    refit!(x) -> Bool

Resize to the current requirements; return false when the existing resource fits.
"""
function refit! end

# Core `emitplan!` walks the passes; backends implement the emission verbs below.
# Interpreted work uses `Immediate`; Metal also uses it for front-of-run updates.

"""
    openrecording(device, plan, passes) -> emitter or nothing

Open an emitter for exactly passes, which may be one partition of the plan.
Return nothing for interpreted execution. Size fixed-capacity recordings for this partition.
"""
openrecording(::Device, ::Plan, ::AbstractUnitRange) = nothing

"""
    closerecording!(emitter, plan) -> recording

Seal the recording and finish its argument-address patch table.
"""
function closerecording! end

"""
    emithead!(emitter, plan)

Emit a global dependency before plan work to cover hazards from previous submissions.
"""
emithead!(e, ::Plan) = nothing

"""
    emitloophead!(emitter, plan, body)

Emit a global dependency and reset iteration timestamps before each repeat body.
Per-pass dependencies do not cover the previous iteration; recording backends must implement this.
"""
function emitloophead! end

"""
    emitbarriers!(emitter, passplan)

Emit the dependencies derived for this pass.
"""
emitbarriers!(e, ::PassPlan) = nothing

"""
    withpredicate(f, emitter, predicate)

Run f with device work discarded when the predicate is zero. `nothing` is unconditional.
"""
withpredicate(f, e, ::Nothing) = f()

"""
    emitupdate!(emitter, plan, passplan)

Apply this run's pending host stores. A reusable recording must not capture their values.
"""
function emitupdate! end

"""
    emitkernel!(emitter, f, args...; ndrange, workgroup_size)

Emit a plain Julia kernel, including core prepare kernels on device-sized dispatch paths.
"""
function emitkernel! end

"""
    emitpreparebarrier!(emitter)

Make prepare writes visible to indirect-command reads and subsequent dispatches.
"""
emitpreparebarrier!(e) = nothing

"""
    workgroupsize(compiled) -> UInt32

Invocations per workgroup of a compiled dispatch: what turns its device-written
item count into the workgroup count of its indirect command.
"""
function workgroupsize end

"""
    emitdispatch!(emitter, dispatch, name)

Emit one compiled dispatch or trace.
"""
function emitdispatch! end

"""
    emitcopy!(emitter, plan, passplan)

Emit a target-to-buffer copy pass.
"""
function emitcopy! end

"""Record a buffer-to-host transfer, including host visibility, in this run."""
function emitreadback! end

"""
    beginrender!(emitter, plan, passplan) -> handle
    drawcompiled!(emitter, handle, draw)
    endrender!(emitter, handle)

Open attachments, emit draws in order, then close the render pass. The handle is
backend-owned; core sets each draw's viewport and bindings on it (`setviewport!`,
`use_bindings!`) before `drawcompiled!` packs its arguments and draws it.
"""
function beginrender! end
function drawcompiled! end
function endrender! end

"""
    compiledraw(c, pass, draw, argoff) -> CompiledDraw
    compile_dispatch(c, dispatch, argoff, indirect) -> compiled dispatch

Compile using core's argument-block offset and indirect-command slot (zero for host-sized work).
Interpreted defaults live in graph/kalaunch.jl.
"""
function compiledraw end
function compile_dispatch end

"""
    passbarriers(c, pass) -> (images, pre, barrier)

The transitions the `Barriers` phase derived for this pass (`pre`), as the backend
lowers them: the image ones one by one ([`imagebarriers`](@ref)), since a layout
change is per image, and the rest into one barrier ([`passbarrier`](@ref)).
"""
function passbarriers(c::Compile, p::Pass)
    g = c.graph
    need = c.prepass[p]
    isimage(t) = t.resource != 0 && resourcekind(g.ids.by_id[t.resource]) isa ImageKind
    return imagebarriers(g.dev, g, filter(isimage, need)), need,
           passbarrier(g.dev, g, filter(!isimage, need))
end

"""
    imagebarriers(device, graph, transitions) -> Vector

A pass's image transitions as the backend emits them, one each. None by default:
a backend that tracks image hazards itself emits no layout changes.
"""
imagebarriers(dev, g, ts::Vector{Transition}) = Nothing[]

"""
    passbarrier(device, graph, transitions)

A pass's other transitions as one barrier the backend emits before it, or `nothing`
by default, for a backend that orders passes without one.
"""
passbarrier(dev, g, ts::Vector{Transition}) = nothing

"""
    recordsplans(device) -> Bool

Whether the device records plans; recordable decides which plans require record! before run!.
"""
recordsplans(::Device) = false

"""
    runscalls(device) -> Bool

Whether a plan replay can execute a call that submits its own work. Interpreted execution
and stream capture can support this; recording a native command buffer alone cannot.
"""
runscalls(::Device) = false

"""
    kernelcompiles(device) -> (; hits, misses)
    resetkernelcompiles!(device)

Report/reset compiled-kernel cache hits and misses. Devices without counters return zeros.

A kernel is compiled once: it is kept by its code, not by the session, so the same code is not
compiled again later in the session (an edit undone) nor in a later one. Every backend's compiler
keeps what it compiles in CompilerCaching's object cache, and counts what it finds there as a hit.
"""
kernelcompiles(::Device) = (; hits = 0, misses = 0)
resetkernelcompiles!(::Device) = nothing

"""
    librarygemm(device, out, A, B, bias, epilogue) -> callable or nothing

Return a callable taking (out, A, B, bias) for the entire fused expression, or nothing
if unsupported. The callable retains its device-owned library handle.
"""
librarygemm(::Device, out, A, B, bias, epilogue) = nothing

"""
    native_conv2d_dispatch!(device, graph, out, x, w;
                            bias=nothing, stride, pad, dilation, groups,
                            epilogue=identity, transposed=false, name)

Declare native convolution; return nothing or (; bias::Bool, epilogue::Bool) for folded
post-operations. Layouts: x=(W,H,Cin,N), w=(KW,KH,Cin/groups,Cout), out=(OW,OH,Cout,N).
Stride/pad/dilation are (x,y), with symmetric padding. Transposed weights are
(KW,KH,Cout/groups,Cin); stride/pad describe the forward convolution. Nonzero output
padding is unsupported: callers decline it before invoking this hook.
"""
native_conv2d_dispatch!(::Device, graph, out, x, w; bias = nothing, stride, pad,
                        dilation, groups, epilogue = identity, transposed = false,
                        name) = nothing

"""
    activationkind(f) -> Symbol

Declare activation semantics beside f: :identity, :relu, :gelu, :gelu_tanh or :unknown.
Backends must decline unknown fusion. Library arithmetic may differ from the Julia implementation.
"""
activationkind(@nospecialize(f)) = :unknown
activationkind(::typeof(identity)) = :identity

"""
    native_gemm_dispatch!(device, graph, out, A, B;
                          bias=nothing, epilogue=identity, name)

Declare native GEMM; return nothing or (; bias::Bool, epilogue::Bool) describing folded
post-operations. The caller declares the remainder; the backend does not plan the graph.
Unlike librarygemm, partial fusion is allowed.
"""
native_gemm_dispatch!(::Device, graph, out, A, B;
                      bias = nothing, epilogue = identity, name) = nothing
# Immediate execution passes a KernelAbstractions backend, declaration a Mantle
# device; backends may answer for either.
native_gemm_available(::Union{Device,KA.Backend}, ::Type, ::Type, ::Type) = false

"""
    native_batched_gemm_dispatch!(device, graph, out, A, B;
                                  transpose_a=false, transpose_b=false,
                                  alpha=1, coldiv=nothing, name)

Declare strided-batched GEMM; return whether handled. The first two axes are matrices;
trailing axes are matching batches without broadcasting. Transposes apply per plane and
alpha scales the product. Optional coldiv=(N,batch...) divides each output column.
Return false when the operands or requested epilogue are unsupported.
"""
native_batched_gemm_dispatch!(::Device, graph, out, A, B;
                              transpose_a = false, transpose_b = false,
                              alpha = 1, coldiv = nothing, name) = false

"""
    native_attention_dispatch!(device, graph, out, q, k, v; scale, name)

Declare fused attention out=softmax(scale·qᵀk)·v without materializing scores.
q/out=(E,Lq,batch...), k/v=(E,Lk,batch...). Operands may be views
(; res,dims,strides,offset), with strides/offset in elements. Return false for unsupported
shapes/views so the caller can materialize operands or declare separate passes.
"""
native_attention_dispatch!(::Device, graph, out, q, k, v; scale, name) = false

"""
    patchable(device) -> Bool

Whether recorded addresses can be patched after storage moves. Otherwise invalidate
and recompile before replay; captured HIP graphs require that path.
"""
patchable(::Device) = true

"""
    openrun(device, plan) -> emitter
    closerun!(device, plan, emitter) -> token
    abandonrun!(device, emitter)
    abandonframe!(device, plan)

Open per-run emission after frame acquisition/refit; close submits and presents, returning
the completion token. On failure abandon the run and any acquired, unpresented frame.
The interpreted default uses Immediate, presents surfaces and returns no token.
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

Release an opened recording after emission fails. Recording backends must implement this;
run emitters and reusable recordings have different ownership and cleanup paths.
"""
abandonrecording!(::Device, e) = nothing
abandonframe!(dev, pl) = nothing

"""
    emitinline!(emitter, region, offset, ptr, nbytes)

Record nbytes from ptr into region at byte offset, carrying the bytes inside the command buffer.
"""
function emitinline! end

"""
    collect!(profiler, device)

Read the last profiled frame's GPU timestamps without waiting ([`readstamps`](@ref))
and sample each pass whose pair is there. A frame still in flight is a skipped
sample, as is one already sampled whose slots the next frame has not rewritten. A
host profiler (no timestamp pool) has nothing to read.
"""
collect!(::Profiler{Nothing}, dev) = nothing
function collect!(prof::Profiler, dev)
    # Core sets `pending` per submission, not per emission: a recording emits its
    # timestamps once but writes them every run.
    prof.pending || return prof
    stamps = readstamps(dev, prof)
    prof.pending = false
    for (i, s) in enumerate(stamps)
        s === nothing && continue
        lo, hi = s
        lo == prof.sampled[i] && continue
        prof.sampled[i] = lo
        ns = elapsed(lo, hi, prof.period_ns)
        ns === nothing || sample!(prof.gpu_ns[i], ns)
    end
    return prof
end

"""
    readstamps(device, profiler) -> Vector{Union{Nothing,Tuple{UInt64,UInt64}}}

Each pass's raw `(start, stop)` timestamps from the profiler's pool, in ticks of its
`period_ns`, or `nothing` where the pair is not available yet. Without waiting.
"""
function readstamps end

"""
    syncbackend(device) -> Backend

Return the backend marker used to lower transitions into stages, access masks and layouts.
"""
function syncbackend end

"""Whether recorded runs and loop iterations start behind a full buffer memory
dependency, including host stores emitted in front of the recording. Core uses
this native fact only for plans that actually take the recorded route."""
buffer_head_ordered(::Any) = false

"""
    profiled!(f, plan, emitter, i)

Emit pass `i` through `f()` and, when the plan profiles, sample its host recording
time and its GPU time between [`passbegin!`](@ref) and [`passend!`](@ref).
"""
function profiled!(f, pl::Plan, e, i::Integer)
    prof = pl.profiler
    prof === nothing && return f()
    begun = passbegin!(pl, e, i)
    t0 = time_ns()
    r = f()
    sample!(prof.host_ns[i], Float64(time_ns() - t0))
    g = passend!(pl, e, i, begun)
    isnan(g) || sample!(prof.gpu_ns[i], g)
    return r
end

"""
    passbegin!(plan, emitter, i) -> begun
    passend!(plan, emitter, i, begun) -> Float64

Open and close the GPU measurement of pass `i`. `passend!` returns its GPU
nanoseconds, or `NaN` when they are read after the run (timestamp queries) or
unknown. The default asks the device ([`gpupasstime!`](@ref)).
"""
passbegin!(::Plan, e, ::Integer) = nothing
passend!(pl::Plan, e, ::Integer, begun) = gpupasstime!(pl.graph.dev)

"""
    storebytes!(emitter, store, offset, ptr, nbytes)

Record an ordered nbytes store into a DeviceArray at byte offset. The backend embeds
or stages the bytes according to its command encoding requirements.
"""
function storebytes! end

"""
    waitfor!(device, token)

Wait for submitted work covered by token. Reject an unsubmitted token with ArgumentError;
waiting does not submit work or substitute a device-wide idle wait.
"""
function waitfor!(dev, tok)
    waitfor(dev, tok) || throw(ArgumentError(
        "waitfor!: token $tok covers work nothing has submitted to this device. A " *
        "token comes from a submission (`run!`, `fence`), and waiting for one " *
        "beyond them would wait for ever."))
    return nothing
end

"""
    waitfor!(plan)

Wait for the last run's token. A plan with no argument memory or no previous run is a no-op.
"""
function waitfor!(pl::Plan)
    am = pl.args
    am === nothing && return nothing
    tok = argtoken(am)      # a method, not a field read: no boxing (see types.jl)
    tok == 0 && return nothing
    passed(pl.graph.dev, tok) && return nothing
    waitfor!(pl.graph.dev, tok)
    return nothing
end


"""
    makeprofiler(device, passes, profile::Bool)

Return nothing when disabled; otherwise create device timestamps or the default host profiler.
"""
makeprofiler(dev::Device, passes, profile::Bool) =
    profile ? hostprofiler(passes) : nothing

"""
    timestamps(device) -> Bool

Whether GPU timestamps can be read without waiting. Such plans measure and partition
recordings using submissionbudget; the default does not provide timestamps.
"""
timestamps(::Device) = false

"""
    submissionbudget(device) -> Float64

Default maximum estimated GPU seconds per submission. Core partitions recordings to
avoid watchdog timeouts and long display stalls; Plan(g; budget) can override it.
"""
submissionbudget(::Device) = 0.1

"""Host-only profiler: no query pool, NaN GPU period/time. Host cost includes recording,
binding, submission and waits; it is not GPU execution time."""
hostprofiler(passes) =
    Profiler(nothing, NaN, 0, [pp.pass.name for pp in passes],
             [Float64[] for _ in passes], [Float64[] for _ in passes],
             zeros(UInt64, length(passes)), false)

"""
    gpupasstime!(device) -> Float64

GPU nanoseconds for the pass just emitted, or NaN when unknown. Profiling-only; a backend
may submit and wait to measure it, serializing an otherwise pipelined frame.
"""
gpupasstime!(::Device) = NaN


"""
    makeargmemory(device, passes) -> ArgMemory or nothing

Lay out argument blocks followed by 256-byte-aligned indirect commands, once per plan.
Keep one copy: packed arguments are not rewritten per run; mutable values use GPURef
addresses. Return nothing when the backend binds arguments directly.
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
    # Built once: the offsets never move, and a view per record would allocate.
    indirect = Any[indirectslot(dev, store, indbase + (k - 1) * INDIRECT_STRIDE) for k in 1:nind]
    return ArgMemory(store, address, ptr, Ref(UInt64(0)), indirect)
end

"""
    argbytes(device, nbytes) -> (store, address, ptr) or nothing

Allocate plan-lifetime host-writable device-addressable bytes; return their owner, device
address and mapped pointer, or nothing for direct argument binding.
"""
argbytes(::Device, nbytes) = nothing

"""
    indirectslot(device, store, offset) -> slot

Return the indirect command at byte offset in store. Prepare kernels write its grid;
slotgrid exposes the three UInt32 workgroup counts.
"""
function indirectslot end

"""
    slotgrid(slot) -> device array of three UInt32

The slot's workgroup counts as a device array. Default: the slot itself.
"""
slotgrid(s) = s


"""
    devicename(device) -> String

Human-readable device name for display/diagnostics. Select behavior using caps, not this string.
"""
function devicename end

# A device prints as its type and name. The default `show` walks its fields, and
# a Metal device's recorded dispatches hold arrays that may be freed by then.
Base.show(io::IO, d::Device) = print(io, nameof(typeof(d)), "(", repr(devicename(d)), ")")

"""
    initbackend!(api)

Initialize/register this build's backend from __init__. Backend-only names stay in its implementation.
"""
function initbackend! end

# Core names a backend under `src/vulkan/` or `src/metal/` may add methods to;
# test/test_backend_vocabulary.jl enforces the list.
const BACKEND_VOCABULARY = (
    :initbackend!,
    :devicename,
    # Runtime hooks outside the graph (runtime/backendhooks.jl).
    :staged_gemm_tile, :videodecodes, :decode_h264_gpu, :h264decoder, :feed!, :decodemore!, :remaining, :copytexels!,
    :isoutofmemory,
    # devices, memory, resources
    :Device, :batchqueue, :capacity, :caps, :maxalloc, :pool,
    # Core answers these for `SubmitChannel`; a backend answers for its own channel
    # type (Metal.jl's `BatchedCommandQueue`).
    :todevice, :reclaim!,
    :bestshape,
    :rawalloc, :rawfree, :constraintof, :mergeconstraints, :compatible, :materialize!,
    :alignment, :bufferusage, :extrausage, :imageusage, :devicearray, :deviceview,
    :deviceslice,
    # A backend answers for its device array, whose offset core cannot see.
    :basealignment,
    :upload!, :download, :devicecopy!, :hostspan, :deviceaddress, :patchable, :release!,
    # host memory a device reads in place, `runtime/api.jl`
    :importhost, :canimporthost,
    :indexbuffer,             # Vulkan needs a usage bit at allocation, Metal does not
    # Driver half of recording reuse; core owns the free list.
    :makerecording, :resetrecording!, :destroyrecording!, :finishrecording!, :recorder,
    # Core owns lifetime decisions; backends provide destructors/stamp storage and
    # recording-specific claims/residency.
    :stampof, :deviceof, :hold!, :holdleaves!,
    # Whether handover pops the hold frame: a one-shot does; a recording submitted
    # every run took its frame when sealed and must not pop another.
    :submissionholds!,
    # Native render-pass primitives shared by hand-recorded and graph emission.
    :compile_draw, :begin_render_pass!, :record_draw!, :end_render_pass!,
    # What `record_draw!` bakes per argument; a device array answers with its
    # address, so a cached plan sees a reallocation.
    :argidentity,
    :setviewport!, :colorimage, :depthimage, :currentimage, :depthformat, :packedsize,
    :storage, :resourcekind, :makeimage, :remakeimage!, :AdaptedAccel, :mipmapped,
    :supports, :supports_graphics, :supports_geometry_stage,
    :supports_tessellation, :supports_mesh_pipeline, :supports_batch_queue,
    :supports_procedural_traversal, :supports_hwtlas, :supports_float64, :supports_int64_atomics,
    :supportspredicate,                       # only whether fixed-size gated work can be discarded
    # Device lifecycle, validation and allocation probes (runtime/lifecycle.jl).
    :debugenv, :validating, :validationmessages!, :submissions, :retired, :tryallocate,
    :devices, :makedevice, :remakedevice, :adopt!,
    # the queue and its tokens; the backend builds core's channel around its driver half
    :SubmitChannel,
    :newchannel, :closechannel!, :submit!, :flush!, :waitidle, :oneshot!,
    :issued, :awaittoken!, :waitfor!, :passed, :fence, :reset_device!, :awaitwrites,
    # sync lowering
    :access, :stages, :layout, :needs_transition, :global_buffer_barriers,
    # Vendor verbs (`vkformat`, `mtlformat`) are backend-private and not listed.
    # A backend names its pixel formats through `nativeformat`; core owns the table
    # (`runtime/format.jl`).
    :nativeformat,
    # compile: the phases are core's, a backend answers these
    :syncbackend,
    :compiledraw, :compile_dispatch, :imagebarriers, :passbarrier,
    :argbytes, :indirectslot,
    # Argument packing is core's (`graph/packing.jl`); a backend adds `passedas`
    # only for its own allocation handles.
    :passedas,
    # What an `llvmcall`'s external symbol does to its pointer; `nothing` means
    # undeclared, which the walk refuses.
    :intrinsic_usage,
    # Library callable types declare their own argument accesses.
    :argument_usage,
    # Properties required from backend-specific compiled dispatches.
    :callgroup,
    :makeprofiler, :Profiler,
    # Whether passes can be timed, so recordings can be cut by the budget
    # (`partitionranges`); and the budget.
    :timestamps, :submissionbudget,
    # Also read from backend-specific compiled dispatches.
    :argsize, :indirectindex,
    # Core defines a default for each; a backend overrides it.
    :retire!, :blocksize, :copy_target!, :gpupasstime!, :passbegin!, :passend!,
    # the walk's primitives
    :openrecording, :closerecording!, :emithead!, :emitloophead!, :emitbarriers!,
    :withpredicate,
    :emitupdate!, :emitdispatch!, :emitcopy!, :emitreadback!, :nativebuffercopy, :readbackbinding, :samereadback,
    :beginrender!, :drawcompiled!, :endrender!,
    :readstamps,
    :emitkernel!, :emitpreparebarrier!, :workgroupsize, :indirectslot, :writegroups!, :slotgrid,
    :storebytes!, :orderstore!,
    :recordsplans, :recordable, :recordablekind, :buffer_head_ordered, :recordspass, :recordsprofiled, :runscalls, :librarygemm, :native_gemm_available,
    :native_conv2d_dispatch!,
    :native_gemm_dispatch!, :native_batched_gemm_dispatch!,
    :native_attention_dispatch!,
    # Array layout for core's GEMM, and split-k scratch for immediate calls.
    :gemmstrides, :splitscratch,
    :kernelcompiles, :resetkernelcompiles!,
    :openrun, :closerun!, :abandonrun!, :abandonframe!, :emitinline!,
    # Submit one baked piece, or give back one never submitted; core owns the
    # partition (`RecordingParts`).
    :submitrecording!, :submitpiece!, :abandonrecording!,
    # graphics verbs, immediate and windowed; a window's are its surface's
    # (`graphics/window.jl`)
    :windowsurface, :fitsurface!, :closesurface!,
    :makeframebuffer, :maketexture, :makesampler, :Surface,
    :acquire_next_image!, :present_frame!,
    :use_bindings!, :bind_textures, :readback_framebuffer, :readback_target,
    :readback_window, :target_extent, :target_format, :target_image, :target_view,
    # Core infers accesses; backends supply adapted argument types, interpreters
    # and stage-wrapper signatures.
    :argtype, :devicebuffertype, :isdevicearray, :resource_identity, :accesscache, :kerneltouches,
    :kernelinterpreter, :kakernelaccesssignature,
    :vertextouches, :fragmenttouches,
    # ray tracing: the native structures under core's builds (`raytracing/api.jl`) and its
    # `BatchedTLAS` (`raytracing/mutation.jl`)
    :build_accel!, :buildnative!, :refitnative!, :refittable, :refitinput, :inputarray,
    :blastype, :nativerecords, :geometryvalue, :instancerecord, :scenevalue,
    # the ray query core's traversal loops (`query_record`, `query_transparent`) drive: device code
    :query_begin, :query_begin_transparent, :query_next!, :candidate_is_box,
    :candidate_primitive_index, :candidate_object_ray, :candidate_ids, :candidate_triangle,
    :commit_triangle!, :commit_intersection!, :committed_kind, :committed_ids,
    :committed_barycentrics, :query_end!,
)
