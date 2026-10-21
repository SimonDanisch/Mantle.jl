# Legacy graph hooks. Core owns scheduling, liveness, placement and barriers;
# backends compile, encode and submit the resulting work. Target verbs: docs/api.md §4.

"""
    isdepth(x) -> Bool

Classify legacy attachment element types: Float32 is depth; other types are color.
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

# Core emitplan! walks passes; backends implement native emission below.
# Interpreted work uses Immediate; Metal also uses it for front-of-run updates.

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

Run f with device work discarded when the predicate is zero. Nothing is unconditional.
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

Return invocations per workgroup. Traces use one because their indirect counts are rays.
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

"""
    beginrender!(emitter, plan, passplan) -> handle
    emitdraw!(emitter, handle, draw)
    endrender!(emitter, handle)

Open attachments, emit draws in order, then close the render pass. The handle is backend-owned.
"""
function beginrender! end
function emitdraw! end
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

Return image-layout changes, transitions and the memory barrier derived for this pass.
The default has no explicit dependencies.
"""
passbarriers(::Compile, ::Pass) = (Nothing[], Transition[], nothing)

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
                            epilogue=identity, name)

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
# Immediate execution carries a KernelAbstractions backend while declaration
# carries a Mantle device.  Backends may answer for either; the default is the
# same absence of a native path.
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
and recompile before replay; captured HIP graphs currently require that path.
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

Read completed timestamp samples without waiting; skip a frame still in flight.
"""
collect!(::Profiler, dev) = nothing

"""
    syncbackend(device) -> Backend

Return the backend marker used to lower transitions into stages, access masks and layouts.
"""
function syncbackend end

"""
    profiled!(f, plan, emitter, i)

Emit pass i and sample its host/GPU cost when profiling. Timestamp backends may override.
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
    # One view per device-sized dispatch, built here because the offsets never
    # move and building one per record is an allocation on the recording path.
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

Return the three UInt32 workgroup counts as a device array.
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

# Legacy backend extension surface, checked by test_backend_vocabulary.jl.
# Replace entries with the target verbs in docs/api.md as their old callers disappear.
const BACKEND_VOCABULARY = (
    # what a backend does once, at load
    :initbackend!,
    # for humans: which GPU this is
    :devicename,
    # Runtime hooks outside the graph (runtime/backendhooks.jl).
    :staged_gemm_tile, :videodecodes, :decode_h264_gpu, :h264decoder, :feed!, :decodemore!, :remaining, :build_blas_aabb, :upload_texture_data!,
    # which of a backend's errors is the device out of memory (the pool's pressure steps)
    :isoutofmemory,
    # devices, memory, resources
    :Device, :backend, :kibackend, :batchqueue, :capacity, :caps, :maxalloc, :pool,
    # A channel names its device. Core answers for its own `SubmitChannel`; a
    # backend whose channel is a type of its driver package's (Metal.jl's
    # `BatchedCommandQueue`) answers for that one, and gives it back to `reclaim!`.
    :todevice, :reclaim!,
    :bestshape,
    :rawalloc, :rawfree, :constraintof, :mergeconstraints, :compatible, :materialize!,
    :alignment, :bufferusage, :extrausage, :imageusage, :devicearray, :deviceview,
    :deviceslice,
    # How far a kernel may trust an operand's base address. Core answers its own
    # resources; a backend answers for its device array, whose offset lives
    # inside it where no core type can see it.
    :basealignment,
    :upload!, :download, :devicecopy!, :hostspan, :deviceaddress, :patchable, :release!,
    # host memory a device reads in place, `runtime/api.jl`
    :importhost, :canimporthost,
    :indexbuffer,
    # The recording primitives a submission channel needs — 2.3. Core owns the
    # free list and when a recording may be reused; these are the driver half.
    :makerecording, :resetrecording!, :destroyrecording!, :finishrecording!, :recorder,
    # Core owns lifetime decisions; backends provide destructors/stamp storage and
    # recording-specific claims/residency.
    :stampof, :deviceof, :hold!, :holdleaves!,
    # Whether handing a recording over pops the channel's hold frame: a one-shot
    # is submitted once and does, a recording that is submitted every run took
    # its frame when it was sealed and must not pop a second one.
    :submissionholds!,
    # Native render-pass primitives shared by hand-recorded and graph emission.
    :compile_draw, :begin_render_pass!, :record_draw!, :end_render_pass!,
    # What `record_draw!` baked about an argument — a backend's device array
    # answers with its address, so a cached plan sees a reallocation.
    :argidentity,
    :setviewport!, :colorimage, :depthimage, :currentimage, :blittarget,                             # Vulkan needs a usage bit at allocation, Metal does not
    :storage, :resourcekind, :makeimage, :remakeimage!, :AdaptedAccel,
    :supports, :supports_graphics, :supports_geometry_stage,
    :supports_tessellation, :supports_mesh_pipeline, :supports_batch_queue,
    :supports_rt_pipeline, :supports_procedural_traversal, :supports_hwtlas, :supports_float64, :supports_int64_atomics,
    :supportspredicate,                       # only whether fixed-size gated work can be discarded
    # Device lifecycle, validation and allocation probes (runtime/lifecycle.jl).
    :debugenv, :validating, :validationmessages!, :submissions, :retired, :tryallocate,
    # the queue and its tokens
    :devices, :defaultdevice!,
    # Core's channel over a backend's queue: the backend constructs it around
    # the driver half it owns, and core never looks inside that.
    :SubmitChannel,
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
    # Library callable types declare their own argument accesses.
    :argument_usage,
    # Properties required from backend-specific compiled dispatches.
    :callgroup,
    :makeprofiler, :Profiler,
    # Whether the profiler can time a pass, and so whether a plan can cut its
    # recording by the budget; and the budget itself (`partitionranges`).
    :timestamps, :submissionbudget,
    # Legacy properties of backend-specific compiled dispatches; remove when the
    # target compiled representation replaces these types.
    :argsize, :devicesized, :indirectindex,
    # Likewise declared rather than left implicit: core defines each of these and
    # a backend overrides it.
    :retire!, :blocksize, :copy_target!, :gpupasstime!,
    # the walk's primitives
    :openrecording, :closerecording!, :emithead!, :emitloophead!, :emitbarriers!,
    :withpredicate,
    :emitupdate!, :emitdispatch!, :emitcopy!, :beginrender!, :emitdraw!, :endrender!,
    # argument_usage is listed once above.
    :profiled!, :collect!,
    :emitkernel!, :emitpreparebarrier!, :workgroupsize, :indirectslot, :writegroups!, :slotgrid,
    :storebytes!,
    :recordsplans, :recordable, :runscalls, :librarygemm, :native_gemm_available,
    :native_conv2d_dispatch!,
    :native_gemm_dispatch!, :native_batched_gemm_dispatch!,
    :native_attention_dispatch!,
    # Core's GEMM asks the backend how its arrays are laid out (`gemmstrides`) and
    # for split-k scratch when the call is immediate (`splitscratch`).
    :gemmstrides, :splitscratch,
    :kernelcompiles, :resetkernelcompiles!,
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
    :use_bindings!, :bind_textures, :transition_image!, :readback_framebuffer, :readback_target,
    :readback_window, :target_extent, :target_format, :target_image, :target_view,
    # Core infers accesses; backends supply adapted argument types, interpreters
    # and stage-wrapper signatures.
    :argtype, :devicebuffertype, :isdevicearray, :accesscache, :kerneltouches,
    :kernelinterpreter, :kakernelaccesssignature,
    :vertextouches, :fragmenttouches, :shadertouches,
    # ray tracing
    :build_accel!, :build_blas, :refit_tlas!, :set_anyhit_pipeline!, :trace_rays!,
    :trace_rays_indirect!, :trace_closest_hits!, :trace_closest_hits_indirect!,
    :trace_closest_hits_anyhit!, :trace_closest_hits_anyhit_indirect!,
)
