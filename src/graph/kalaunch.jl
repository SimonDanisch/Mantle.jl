# Running a compiled graph on a KernelAbstractions backend.
#
# Every backend whose kernels are KA kernels executes a plan the same way: bake
# one callable per dispatch at compile time, then a frame is a loop over
# callables. The host and Metal backends share ALL of it — neither overrides a
# single function here, because `storage` already knows how to turn each
# resource kind into the array a kernel takes.
#
# This file was `src/host/host.jl`. Moving it here is what makes the Metal
# backend's compute path about twenty lines instead of a second copy of it, and
# it is the shape any future CUDA or ROCm backend gets for free.
#
# The Vulkan backend does NOT use this. It records commands into a command
# buffer rather than closing over Julia callables, so it defines its own
# `run!(::Pipelines, ::Compile{LavaDevice})` — which is why `Compile` carries
# its device as a type parameter: a backend that needs a different execution
# model says so by being more specific than the default below.

"""
    resolve(device, x)

The value a baked launch should pass for the graph argument `x`.

Defaults to [`storage`](@ref), which already answers this for every resource
kind: a `Buffer`'s storage is `deviceview(dev, store)` — the backend's array
over its pool region — and a materialised transient's is the array `Place` gave
it. Neither the host nor the Metal backend overrides this, and the hook only
still exists so one with a different notion of "the array a kernel takes" can.
"""
resolve(::Device, x) = storage(x)

"""
One launch, resolved as far as a launch can be.

`K`, `A` and `N` are concrete, so the call inside is direct: no lookup and no
splat over an abstract tuple. The one thing left for run time is the one that
means nothing otherwise — a device-computed count.
"""
struct Launch{K,A<:Tuple,N,D}
    kernel::K
    args::A
    ndrange::N
    # Carried so a deferred count can ask the right backend to settle before it
    # is read — see `ndrangeof`.
    device::D
end

"""
A count that is read at launch, not at compile.

The point of a device-computed count is that an earlier pass writes it; reading
it at compile would capture the value from before that pass ran.
"""
struct DeferredRange{S}
    store::S
end

ndrangeof(::Device, n) = n

"""
    ndrangeof(device, r::DeferredRange) -> Int

Read a device-computed dispatch count, on the host, at launch.

**This reads memory an earlier kernel wrote, so the earlier kernel has to have
finished.** On the host backend that is free: the previous step ran to
completion on this thread before this one started. On a GPU backend it is not —
the producing launch is queued, the host read races it, and the count comes back
stale.

That race is exactly what a wavefront path tracer trips over: Hikari dispatches
each queue over `DeviceRange(queue.size)`, so a stale count launches too few
threads and the frame silently loses whatever those threads would have shaded.
It shows up as pixels that are black in one run and lit in the next, not as an
error — 4,197 of 16,384 pixels differing run to run on an Apple M5 before this
`awaitwrites` was here.

The default therefore synchronises. A backend whose reads cannot race — the host
— says so by overriding [`awaitwrites`](@ref) to do nothing.
"""
function ndrangeof(dev::Device, r::DeferredRange)
    awaitwrites(dev)
    return Int(first(r.store))
end

"""
    awaitwrites(device)

Ensure device work already submitted has completed, before the host reads memory
it wrote.

Defaults to `synchronize`, because on any backend that queues work a host read
of device memory races whatever wrote it. The host backend overrides this to a
no-op: its "device" writes happen on the calling thread.
"""
awaitwrites(dev::Device) = KernelAbstractions.synchronize(backend(dev))

# A count of zero launches nothing, which is what it means and what a recording
# backend does with it — an indirect dispatch of zero workgroups is a no-op
# there. Without this, KA gets an empty ndrange, and a wavefront round whose
# queue emptied is the ordinary case rather than an edge.
function (l::Launch{K,A,N})() where {K,A,N}
    nd = ndrangeof(l.device, l.ndrange)
    nd == 0 && return nothing
    l.kernel(l.args...; ndrange = nd)
    return nothing
end

kernelfor(k, ::Nothing, backend) = k(backend)
kernelfor(k, group, backend) = k(backend, group)

# A `Launch` resolved its count at `bake`: a `DeviceRange` with a ceiling became
# the ceiling, one without became a `DeferredRange`. Neither is sized on the
# device the way a recording backend means it — see `PassPlan.indirect`.
devicesized(::Launch) = false

# A function barrier: `args` is concrete inside `Launch`, which is what makes
# the call in `(l::Launch)()` specialise.
function bake(c::Compile, d::Dispatch)
    dev = c.graph.dev
    Launch(kernelfor(d.kernel, d.group, backend(dev)),
           map(a -> resolve(dev, a), d.args), bakedrange(c, d.ndrange), dev)
end

bakedrange(::Compile, n) = n

# A `DeviceRange` with a ceiling does NOT need the host to read its count.
#
# `dispatchrange` on the recording path already compiles the kernel against
# `something(r.max, INDIRECT_CEILING)` and lets the kernel's own `i <= n` check
# do the bounding — so every kernel dispatched over a `DeviceRange` is required
# to bounds-check itself, repo-wide, and over-dispatching it is defined to be a
# no-op for the surplus threads.
#
# This path did not use that. It built a `DeferredRange`, whose `ndrangeof`
# calls `awaitwrites` — a full device synchronise, on the host, before EVERY
# such dispatch. Measured on an M5, killeroo-gold at 684x513/8spp: 320 of them
# per frame, 0.585 s of a 0.602 s frame. 97 % of the render was the host
# standing still waiting for a number it then used to launch the next kernel,
# and hardware traversal made no difference because traversal was not the cost.
#
# Ordering does not depend on the sync: the producing and consuming dispatches
# go to the same queue, which runs them in order. The sync existed only so the
# HOST could read a device-written count. Take the host out and it is not
# needed.
#
# `max === nothing` still reads it — there is no ceiling to dispatch instead,
# and guessing one would launch `INDIRECT_CEILING` threads over a queue that
# might hold four.
bakedrange(c::Compile, r::DeviceRange) =
    r.max === nothing ? DeferredRange(resolve(c.graph.dev, r.count)) : r.max

"""
One `Pipelines` phase, for every backend: lay the passes out and have the
backend compile each draw and dispatch against that layout.

What is decided here is Mantle's — the argument block each draw and dispatch
owns (`argcursor`), the indirect-command slot each device-sized dispatch owns
(`indcursor`), that every attachment of a render pass covers one render area —
and what a compiled draw or dispatch IS is the backend's, through
`compiledraw` and `compile_dispatch`. An interpreted backend answers with a
callable and an argument size of zero, a recording backend with the pipeline
and the size of the block it will pack. The Vulkan backend used to run this
whole phase itself, cursor and all, beside this one.
"""
function run!(::Pipelines, c::Compile)
    argcursor = 0        # every draw's and dispatch's argument block, laid out once
    # …and every device-sized dispatch's indirect command beside them. Counted
    # here rather than allocated at record because the plan knows how many it
    # has, which is the same reason `argcursor` is here.
    indcursor = 0
    for p in ordered(c)
        checkrenderarea(p)
        cds = CompiledDraw[]
        for d in p.draws
            cd = compiledraw(c, p, d, argcursor)
            # How many pipelines the draws resolved to — two draws sharing one
            # is the thing worth counting, which is why a dispatch's is not here.
            push!(c.pipelines, cd.compiled)
            push!(cds, cd)
            argcursor += argalign(argsize(cd))
        end
        # `Any[]` and then narrowed: a pass declares `Dispatch`es or `Trace`s,
        # `compile_dispatch` answers both, and the two compile to different
        # types. `identity.` gives back a concrete vector when the pass holds
        # one kind — every pass in tree — so the per-frame loop specialises.
        compiled = Any[]
        for d in p.dispatches
            ind = d.ndrange isa DeviceRange ? (indcursor += 1) : 0
            cd = compile_dispatch(c, d, argcursor, ind)
            push!(compiled, cd)
            argcursor += argalign(argsize(cd))
        end
        imgs, pre, barrier = passbarriers(c, p)
        push!(c.passes, PassPlan(p, cds, identity.(compiled), imgs, pre, barrier))
    end
    return c
end

"""
One render area for the pass: attachments that disagree about their size would
silently render into part of the larger one. Checked once at compile rather
than per frame.
"""
function checkrenderarea(p::Pass)
    p.kind === :render || return nothing
    want = target_extent(first_target(p))
    for t in (p.targets..., p.depth)
        t === nothing && continue
        target_extent(t) == want ||
            throw(ArgumentError("pass \"$(p.name)\": attachments are $(want) and " *
                                "$(target_extent(t)); every attachment shares one render area"))
    end
    return nothing
end

"""The bytes a compiled draw or dispatch packs into the plan's argument memory.
Zero for a backend that binds arguments directly: an interpreted dispatch is a
`Launch`, and its draw records nothing."""
argsize(d::CompiledDraw) = d.argsize
argsize(d::CompiledDispatch) = d.argsize
argsize(d::CompiledTrace) = d.argsize
argsize(::Launch) = 0

"""The indirect-command slot a compiled dispatch owns; zero for one sized on the
host, and for every interpreted launch."""
indirectindex(d::CompiledDispatch) = d.indirect
indirectindex(d::CompiledTrace) = d.indirect
indirectindex(::Launch) = 0

# ── Device-sized dispatches ──────────────────────────────────────────────────
#
# A dispatch over a `DeviceRange` reads its count on the device, so something on
# the device has to turn that count into the indirect command the dispatch
# reads. This is that something, and it is Mantle's: which dispatches, what
# they divide by, and what a `repeat!` gate does to them are graph facts. A
# backend launches the kernel (`emitkernel!`) and puts the barrier after it
# (`emitpreparebarrier!`).

"""
One fused prepare for every dispatch and trace in this pass that sizes itself
on the device, and one barrier behind all of them.

A device-sized dispatch costs a prepare and a barrier the command processor's
read of the count depends on, and that barrier cannot be derived away — the
prepare is Mantle's own machinery rather than a pass, so nothing declared it.
Left per dispatch it serialises a pass's dispatches whatever the schedule said
they could do; fused, it is one kernel writing every count, one barrier, then
the dispatches back to back.

The pass's `repeat!` gate is folded in: a discarded iteration writes ZERO groups
for every one of its dispatches, so a backend with no way to discard recorded
work still runs nothing for it. That is what lets `repeat!` be portable; a
backend that can also skip the commands themselves does so in `withpredicate`.
Emitted by `emitpass!` BEFORE the predicate scope, because the fold is only
worth anything if the prepare runs for the discarded iteration too.
"""
function emitprepares!(e, pl::Plan, pp::PassPlan)
    pp.indirect || return nothing
    am = pl.args
    inds = Any[]
    counts = Any[]
    wss = UInt32[]
    for d in pp.dispatches
        k = indirectindex(d)
        k == 0 && continue
        push!(inds, indirectof(am, k))
        push!(counts, storage(d.ndrange.count))
        push!(wss, workgroupsize(d))
    end
    n = length(inds)
    n == 0 && return nothing
    pred = pp.pass.predicate
    gate = pred === nothing ? nothing : (storage(pred[1]), Int32(pred[2]))
    emitkernel!(e, multi_prepare_indirect_kernel,
                ntuple(i -> inds[i], n), ntuple(i -> counts[i], n),
                ntuple(i -> wss[i], n), gate;
                ndrange = 1, workgroup_size = (1, 1, 1))
    emitpreparebarrier!(e)
    return nothing
end

# One thread writes every one of this pass's indirect commands. Statically
# unrolled over the tuples; compiled per arity via the normal kernel cache.
@inline multiprepare!(::Tuple{}, ::Tuple{}, ::Tuple{}, go) = nothing
@inline function multiprepare!(inds::Tuple, sizes::Tuple, wss::Tuple, go)
    n = UInt32(@inbounds sizes[1][1])
    ws = wss[1]
    groups = go ? (n + ws - UInt32(1)) ÷ ws : UInt32(0)
    @inbounds begin
        inds[1][1] = groups      # groupCountX, or the ray count of a trace
        inds[1][2] = UInt32(1)
        inds[1][3] = UInt32(1)
    end
    multiprepare!(Base.tail(inds), Base.tail(sizes), Base.tail(wss), go)
    return nothing
end

"""Whether the pass's `repeat!` iteration runs: no gate, or a nonzero flag."""
@inline gateopen(::Nothing) = true
@inline gateopen(gate::Tuple) = (@inbounds gate[1][gate[2] + Int32(1)].go) != UInt32(0)

function multi_prepare_indirect_kernel(inds, sizes, wss, gate)
    multiprepare!(inds, sizes, wss, gateopen(gate))
    return nothing
end

# A trace's indirect command counts rays, not workgroups.
workgroupsize(::CompiledTrace) = UInt32(1)

# The interpreted answers. A draw is compiled against the attachments it will
# hit — the formats come from the pass's own targets, because a pipeline is
# compiled FOR a set of attachments, and taking them from anywhere else is how a
# pipeline ends up compiled for a target it is never drawn into — and a
# dispatch is the `Launch` above.
function compiledraw(c::Compile, p::Pass, d, argoff::Int)
    dev = c.graph.dev
    color_formats = [attachment_format(t) for t in p.targets]
    depth_format  = p.depth === nothing ? nothing : attachment_format(p.depth)
    vargs = map(a -> resolve(dev, a), d.args)
    fargs = map(a -> resolve(dev, a), d.frag_args)
    compiled = compile_draw(dev, d.shader, color_formats, depth_format, vargs, fargs)
    return CompiledDraw(compiled, d.shader, (vargs..., fargs...), bakedcount(c, d.count), argoff, 0)
end
compile_dispatch(c::Compile, d::Dispatch, ::Int, ::Int) = bake(c, d)

"""What a draw's count resolves to once storage exists."""
bakedcount(::Compile, n) = n
bakedcount(c::Compile, n::Commands) = Commands(resolve(c.graph.dev, n.resource))

"""The element type an attachment is, which is the format a pipeline compiles for."""
attachment_format(t) = eltype(t)

# Where a host store lands on a KernelAbstractions backend: the plain in-place
# write, on the calling thread, at the update pass's position. Exact on the host
# backend, where a step runs to completion before the next starts; on Metal a
# host write into memory the previous run's command buffer may still read, which
# is `awaitwrites`' question and not this one's. `landstores!` decides WHEN a
# store is safe to take out of its cell; this is only WHERE it goes.
landhost!(r::GPURef{T}, ptr::Ptr{T}) where {T} = (update!(r, unsafe_load(ptr)); nothing)
landhost!(b::Buffer, data::AbstractVector, from::Integer) =
    (update!(b, from:(from + length(data) - 1), data); nothing)

"""
    run!(plan)

Run `plan` once. The sequence is core and the same on every backend: check the
plan is live, reclaim what was dropped, let the backend poll and sync its
windows ([`beforeframe!`](@ref)), refit any surface-tracking transients, check
the attachment extents still agree, then hand the plan to the backend
([`execute!`](@ref)).

`execute!` is what a backend IS: the Vulkan device submits the plan's recording
in one submission; a KernelAbstractions backend loops the baked passes. `run!`
records nothing and writes no argument memory — a value that changes between
runs is a [`GPURef`](@ref) whose store lands as a command in this run's own
submission.
"""
function run!(pl::Plan)
    checklive(pl, pl.slabs, length(pl.graph.transients))
    # Anything dropped without a `free!` goes back here, one submission boundary
    # after it was dropped. Cheap and a no-op when nothing was.
    reclaim!(pool(pl.graph.dev), pl.graph.dev)
    # The backend's per-frame preamble: poll the window, bring its swapchain up
    # to date, take this frame's image, and read back the profiler's last
    # frame. Nothing for an offscreen graph on any backend.
    beforeframe!(pl.graph.dev, pl)
    # From here the frame holds an image, and a failure hands it back untouched
    # (`abandonframe!`) so the next frame can acquire.
    try
        # A surface that resized moved every attachment that follows it; refit
        # here, AFTER the acquire and before anything is recorded: the acquire
        # is where a resize is noticed last (the presentation engine reports it
        # there, and the framebuffer size is asked again), and a frame refit
        # before it was recorded against a swapchain the acquire then rebuilt —
        # a colour target at the new size beside a depth target at the old one,
        # which is a lost device on NVIDIA. A frame where nothing moved costs one
        # size compare per tracking transient, and `refit!` returns `false` for
        # a plan that follows nothing.
        refit!(pl)
        checkextents(pl)
        # A recording that a move or a refit threw away is written again here,
        # on the owning thread, before it is submitted. Not a first recording: a
        # plan that was never recorded is refused by the backend's `execute!`.
        pl.stale && (pl.stale = false; record!(pl))
        execute!(pl.graph.dev, pl)
    catch
        abandonframe!(pl.graph.dev, pl)
        rethrow()
    end
    return nothing
end

"""
    beforeframe!(device, plan)

The per-frame preamble, before anything is refit or emitted: read back the
profiler's last frame, bring every surface's window up to date (poll it, sync
its swapchain — `beginframe!`) and take the image this frame draws into. The
acquire is here, ahead of `refit!`, because it is the last place a resize can
be noticed; whatever it notices is what the frame is then refit for. Failing
before the acquire leaves nothing behind.
"""
function beforeframe!(dev, pl::Plan)
    # Before this run overwrites them, and without waiting: see `collect!`.
    pl.profiler === nothing || collect!(pl.profiler, dev)
    for s in pl.graph.surfaces
        beginframe!(s.win)
        acquire_next_image!(s.win)
    end
    return nothing
end

"""
    execute!(device, plan)

One run of the plan, the same on every backend: open the run, put in it what
this run has to say, close it. A plan with a recording says only its host
stores and address patches — in a FRONT the backend opens only when there is
something to put in it, so a run with nothing pending hands over exactly the
recording — and the recording follows in the same submission. A plan without
one is walked into the run, pass by pass: an interpreted backend always, a
windowed plan on any backend until step 9 records those too.

Never records. A recordable plan on a recording backend that was not
`record!`ed is refused here.
"""
function execute!(dev, pl::Plan)
    rec = pl.recording
    tok = if rec === nothing
        recordable(pl) && recordsplans(dev) && throw(ArgumentError(
            "run!: this plan has not been recorded. Call `record!(plan)` once after " *
            "building it; `run!` submits that recording and never records."))
        e = openrun(dev, pl)
        try
            emit!(e, pl)
        catch
            abandonrun!(dev, e)
            rethrow()
        end
        closerun!(dev, pl, e)
    else
        front = anydirty(pl.hostwritten) || !isempty(pl.pending_patches) ?
                openrun(dev, pl) : nothing
        if front !== nothing
            try
                emitupdates!(front, pl)
                emitpatches!(front, pl)
            catch
                abandonrun!(dev, front)
                rethrow()
            end
        end
        closerun!(dev, pl, front)
    end
    am = pl.args
    am === nothing || setargtoken!(am, tok)
    # A frame's timestamps are pending from the submission that writes them.
    pl.profiler === nothing || (pl.profiler.pending = true)
    return nothing
end

"""
The update passes alone, in front of a recording: the host stores waiting on
the plan's resources, as commands in this run's own submission, behind the
barriers the pass derived. The emitter lands them (`emitupdate!`); a recording
never holds one, which is why they are here and not in it.
"""
function emitupdates!(e, pl::Plan)
    for pp in pl.passes
        pp.pass.kind === :update || continue
        emitupdate!(e, pl, pp)
    end
    return nothing
end

"""
The pending address patches, in front of a recording — the run half of
`notify_move!`: eight bytes per moved pointer, inline in the command buffer,
the same command a `GPURef` store is.
"""
function emitpatches!(e, pl::Plan)
    patches = lock(pool(pl.graph.dev).lock) do
        p = pl.pending_patches
        # A run with nothing pending allocates nothing — the empty check is
        # under the lock because the append happens under it (notify_move!).
        isempty(p) ? nothing : (pl.pending_patches = Tuple{Region,Int,UInt64}[]; p)
    end
    patches === nothing && return nothing
    for (r, off, val) in patches
        ref = Ref(val)
        GC.@preserve ref emitinline!(e, r, off, Base.unsafe_convert(Ptr{Cvoid}, ref), 8)
    end
    return nothing
end

"""What `landstores!` hands a plan's stores to: `emitstore!` bound to one emitter."""
struct StoreEmitter{E}
    e::E
end
(s::StoreEmitter)(r::GPURef{T}, ptr::Ptr{T}) where {T} = emitstore!(s.e, r, ptr)
(s::StoreEmitter)(b::Buffer, data::AbstractVector, from::Integer) = emitstore!(s.e, b, data, from)

"""
    emitstore!(emitter, ref, ptr)
    emitstore!(emitter, buffer, data, from)

One host store as a command in this run: a `GPURef`'s straight out of its
pending cell — `ptr` names the cell, so no copy of the value exists on the way
and the camera's 736 bytes cross without a box — and a `Buffer`'s range from
the vector `update!` kept. Both are bytes into the resource's store at an
element offset, which is all a backend is asked to carry: `storebytes!`.
"""
emitstore!(e, r::GPURef{T}, ptr::Ptr{T}) where {T} =
    storebytes!(e, r.store, 0, Ptr{Cvoid}(ptr), sizeof(T))
function emitstore!(e, b::Buffer{T}, data::AbstractVector{T}, from::Integer) where {T}
    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src storebytes!(e, b.store, (Int(from) - 1) * sizeof(T),
                                 Ptr{Cvoid}(pointer(src)), length(src) * sizeof(T))
    return nothing
end

"""
    timings(plan) -> Vector{PassTiming}

What each pass cost, as a median over the last [`NSAMPLES`](@ref) frames.
`gpu_ms` is `NaN` on a backend without timestamps; the host side is what
`run!` itself can see.
"""
function timings(pl::Plan)
    prof = pl.profiler
    prof === nothing &&
        throw(ArgumentError("this plan was not built to be profiled; use Plan(g; profile = true)"))
    collect!(prof, pl.graph.dev)
    med(x) = isempty(x) ? NaN : (q = sort(x); q[(length(q) + 1) ÷ 2])
    return [PassTiming(prof.names[i], pl.passes[i].pass.kind,
                       med(prof.host_ns[i]) / 1e6, med(prof.gpu_ns[i]) / 1e6,
                       max(length(prof.host_ns[i]), length(prof.gpu_ns[i])))
            for i in eachindex(prof.names)]
end

# ── The pass walk, and recording it ──────────────────────────────────────────
#
# One walk over a plan's passes, in the order the compile settled on, spoken
# through the primitives declared in `graph/backend.jl`. `record!` walks it once
# into a backend's recording; a backend without recordings walks it on every
# `execute!` with the `Immediate` emitter below, whose primitives do the work on
# the spot. The Vulkan backend carried a second copy of this walk — `emit!`,
# `emitpass!`, `emitwork!` — with `record!` hung off it, which put the
# sequence in a backend and left core with a stub.

"""
    emit!(emitter, plan)

Walk the plan's passes into `emitter`: the head barrier, then every pass —
its barriers, its predicate scope and its work — in compile order.
"""
function emit!(e, pl::Plan)
    emithead!(e, pl)
    # Two loops rather than one with a branch in it: a frame that is not being
    # profiled must not pay a comparison per pass.
    if pl.profiler === nothing
        for pp in pl.passes
            emitpass!(e, pl, pp)
        end
    else
        for (i, pp) in enumerate(pl.passes)
            profiled!(pl, e, i) do
                emitpass!(e, pl, pp)
            end
        end
    end
    return nothing
end

"""
One pass: its barriers, its predicate scope, its work.

An update pass IS its stores, and whether they can land where the walk is (an
interpreted run, a one-shot) or have to wait for the run that has the values
(a recording) is the emitter's answer, so the emitter takes the whole pass.
"""
function emitpass!(e, pl, pp::PassPlan)
    p = pp.pass
    p.kind === :update && return emitupdate!(e, pl, pp)
    emitbarriers!(e, pp)
    # The fused prepare comes BEFORE the predicate scope, so it runs whether or
    # not the iteration does: it folds the gate in and writes zero rays or
    # groups for a discarded one. Inside the scope it was discarded with the
    # iteration — and a trace-rays command is not subject to conditional
    # rendering (VK_KHR_ray_tracing_pipeline leaves it out), so the trace of a
    # discarded bounce still ran, with a ray count read from a slot nothing had
    # written. `test_discarded_iteration_prepare.jl` pins the slot.
    p.kind === :compute && emitprepares!(e, pl, pp)
    # The predicate scope opens AFTER the barriers and the prepare, and closes
    # before the next pass's, so a discarded iteration still orders the ones
    # around it.
    withpredicate(e, p.predicate) do
        emitwork!(e, pl, pp)
    end
    return nothing
end

"""The pass's own commands, once its barriers, its prepare and its predicate
are dealt with."""
function emitwork!(e, pl::Plan, pp::PassPlan)
    p = pp.pass
    if p.kind === :compute
        for d in pp.dispatches
            emitdispatch!(e, d, p.name)
        end
    elseif p.kind === :copy
        emitcopy!(e, pl, pp)
    else
        h = beginrender!(e, pl, pp)
        try
            for d in pp.draws
                emitdraw!(e, h, d)
            end
        finally
            # Closed even if a draw throws: a pass left open takes the whole
            # command buffer with it, and the next frame's failure would name
            # the wrong pass.
            endrender!(e, h)
        end
    end
    return nothing
end

"""
    record!(plan) -> plan

Write the plan's commands once, so every `run!` hands the same command buffer
to the device instead of rebuilding it.

The precondition is that every device address the recording names is the same
next time. A plan gives that outright: placement is fixed, the arena is the
device's, a plan holds references to everything it names, and an `Update`
writes its target in place rather than replacing it. An argument that has to
change between runs is a [`GPURef`](@ref): the commands are packed with its
device address, which does not move, and the value behind it is written by an
`Update` in the run's own submission. Everything else is resolved HERE and is
that value for the plan's life.

Call it once after building the plan. `run!` submits the recording and never
records — a plan that was never recorded is refused there, not recorded on the
way past, so the cost lands where it was asked for and nowhere else. A backend
with no command buffers to build has nothing to record and gets the plan back
unchanged; `run!` walks its passes each time instead.

Not yet for plans with a surface: a swapchain image is a different image every
frame and a recording names one. Headless plans have no such thing.

It does NOT run the plan. `bake!`, which this replaced, did.
"""
function record!(pl::Plan)
    pl.recording === nothing || return pl
    recordable(pl) || throw(ArgumentError(
        "record!: this plan draws to a surface. A swapchain image is a different " *
        "image every frame and a recording names one, so a windowed plan needs a " *
        "recording per swapchain image — which is not built yet. Headless plans " *
        "record today."))
    e = openrecording(pl.graph.dev, pl)
    e === nothing && return pl
    # Where every device address the pack writes lands, keyed by the address —
    # derived once, from this one pack, by `closerecording!`; consulted only by
    # `notify_move!`, never per run.
    empty!(pl.patchtab)
    empty!(pl.pending_patches)
    emit!(e, pl)
    pl.recording = closerecording!(e, pl)
    listen_moves!(pool(pl.graph.dev), pl)
    return pl
end

"""Whether this plan's commands can be written once, or have to be emitted per
run. One reason left, and step 9 removes it: a plan drawing to a SURFACE names a
swapchain image inside its recorded commands, and that is a different image
every frame."""
recordable(pl::Plan) = isempty(pl.graph.surfaces)

# ── The immediate emitter ────────────────────────────────────────────────────
#
# What a backend without recordings walks with: every primitive does its work
# right now, through the interpreted verbs in `graphics/commands.jl` and the
# callables the compile made of the dispatches.

function withpredicate(f, ::Immediate, (pred, i)::Tuple{Any,Int})
    # The flag is an array the gate dispatch just wrote on this thread, so
    # discarding the iteration is reading it and not running the body.
    storage(pred)[i + 1].go == 0 && return nothing
    return f()
end

# Landing the stores the plan's resources are holding IS the update pass's
# body, and it happens at the pass's graph position — before every pass that
# declared a read, which is where `hostwritten!` reserved a `CopyDst` for it.
emitupdate!(::Immediate, pl::Plan, ::PassPlan) =
    (landstores!(landhost!, pl.hostwritten); nothing)

emitdispatch!(::Immediate, d, ::AbstractString) = (d(); nothing)

function emitcopy!(::Immediate, pl::Plan, pp::PassPlan)
    p = pp.pass
    copy_target!(pl.graph.dev, storage(p.dst), first_target(p))
    return nothing
end

function beginrender!(::Immediate, pl::Plan, pp::PassPlan)
    p = pp.pass
    # `target_view`, not `storage`: an attachment is bound by whatever the
    # backend hands a render pass, and for a transient that is its view rather
    # than the arena it was placed in.
    targets = Any[target_view(t) for t in p.targets]
    depth = p.depth === nothing ? nothing : target_view(p.depth)
    return begin_render_pass!(pl.graph.dev, targets, p.loads, depth, p.depth_load)
end
emitdraw!(::Immediate, handle, d) =
    (record_draw!(handle, d.compiled, d.args, d.count); nothing)
endrender!(::Immediate, handle) = (end_render_pass!(handle); nothing)
