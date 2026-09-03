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

A `Ref` is deliberately NOT resolved — see below.
"""
resolve(::Device, x) = storage(x)

# A `Ref` is resolved per run rather than at compile — see `argvalue`. Keeping
# it in `args` and reading it at launch is what makes it mean anything: resolved
# here it would be the value the plan was built with, for ever.
resolve(::Device, x::Base.RefValue) = x

"""
One launch, resolved as far as a launch can be.

`K`, `A` and `N` are concrete, so the call inside is direct: no lookup and no
splat over an abstract tuple. The two things left for run time are the ones that
mean nothing otherwise — a `Ref`'s current value and a device-computed count.
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
    l.kernel(map(argvalue, l.args)...; ndrange = nd)
    return nothing
end

kernelfor(k, ::Nothing, backend) = k(backend)
kernelfor(k, group, backend) = k(backend, group)

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
Bake the steps.

LAST in the phase order, and that is a constraint rather than a convention:
[`resolve`](@ref) reads a transient's storage, which does not exist until
`Place` has given it an offset. Everything a frame could have decided is decided
here — the kernel is specialised, the arguments are bound, the ndrange is fixed
— so a run is a loop over callables and nothing else.

One `PassPlan` per pass, holding callables where a recording backend holds
recorded commands. That is the same slot, which is why the plan type is shared:
`Barriers` produced nothing on this path, so `images` and `pre` are empty and
the barrier is `nothing`.
"""
function run!(::Pipelines, c::Compile)
    for p in ordered(c)
        steps = Any[bake(c, d) for d in p.dispatches]
        push!(c.passes, PassPlan(p, bakedraws(c, p), steps,
                                 Nothing[], Transition[], nothing))
    end
    return c
end

"""
    bakedraws(c, pass) -> Vector{CompiledDraw}

Compile a render pass's draws, once, against the attachments they will hit.

Empty for anything that is not a render pass, which is what a compute-only graph
consists of and why this used to push `CompiledDraw[]` unconditionally.

The formats come from the pass's own attachments rather than from the caller: a
pipeline is compiled FOR a set of attachments, and taking them from anywhere
else is how a pipeline ends up compiled for a target it is never drawn into.
"""
function bakedraws(c::Compile, p::Pass)
    (p.kind === :render && !isempty(p.draws)) && return _bakedraws(c, p)
    return CompiledDraw[]
end

function _bakedraws(c::Compile, p::Pass)
    dev = c.graph.dev
    color_formats = [attachment_format(t) for t in p.targets]
    depth_format  = p.depth === nothing ? nothing : attachment_format(p.depth)
    out = CompiledDraw[]
    argoff = 0
    for d in p.draws
        vargs = map(a -> resolve(dev, a), d.args)
        fargs = map(a -> resolve(dev, a), d.frag_args)
        compiled = compile_draw(dev, d.shader, color_formats, depth_format,
                                vargs, fargs)
        # `argoff`/`argsize` are the recording backends' argument-memory slots.
        # A backend that binds per draw has no such memory, and zero is the
        # honest value rather than a made-up offset into something that does not
        # exist.
        push!(out, CompiledDraw(compiled, d.shader, (vargs..., fargs...),
                                bakedcount(c, d.count), argoff, 0))
    end
    return out
end

"""What a draw's count resolves to once storage exists."""
bakedcount(::Compile, n) = n
bakedcount(c::Compile, n::Commands) = Commands(resolve(c.graph.dev, n.resource))

"""The element type an attachment is, which is the format a pipeline compiles for."""
attachment_format(t) = eltype(t)

"""
Land one pending `Update`.

`UpdateRef` and `applyupdates!` are core: they decide WHEN a write is safe to
take out of the box. This is the other half, WHERE it goes, and on a
KernelAbstractions backend it is the plain in-place write — the memory is
directly addressable, so the resource keeps its store.

The Vulkan route renames into a fresh store instead (`write_update!` in
`src/vulkan/graph.jl`), because there the copy is RECORDED rather than executed
and the GPU may still be reading the old bytes. Same contract, different cost.
"""
function writeupdate!(r::UpdateRef, data::AbstractVector)
    isempty(data) && return nothing
    dst = r.resource
    r.range === nothing ? update!(dst, data) : update!(dst, r.range, data)
    return nothing
end

"""
One frame: call the baked steps in the order compile settled on.

No synchronisation, no lookup, no branch. Everything that could be decided was.
"""
function run!(pl::Plan)
    checklive(pl, pl.slabs, length(pl.graph.transients))
    reclaim!(pool(pl.graph.dev), pl.graph.dev)
    # A graph that reaches a window: poll and acquire BEFORE any pass, present
    # after the last one. The order is the whole contract — see
    # `graphics/commands.jl`. `g.surfaces` is empty for every offscreen graph,
    # so this costs one `isempty` there.
    surfaces = pl.graph.surfaces
    for s in surfaces
        beginframe!(s.win)
    end
    # Between polling and acquiring, and nowhere else. A surface that resized
    # moved every attachment that follows it, and the plan's placement was for
    # the old size; recompiling here leaves nothing behind, while the same
    # discovery made during recording would leave a half-recorded frame holding
    # offsets into freed storage.
    #
    # Costs one size compare per tracking transient in a frame where nothing
    # moved, and `refit!` returns `false` for every transient that follows
    # nothing at all.
    isempty(surfaces) || refit!(pl)
    for s in surfaces
        acquire_next_image!(s.win)
    end
    prof = pl.profiler
    # Two loops rather than one with a branch in it: a frame that is not being
    # profiled must not pay a comparison and a `time_ns` per pass, and the
    # profiled one wants the clock as close to the pass as it can get.
    if prof === nothing
        for pp in pl.passes
            runpass!(pl, pp)
        end
    else
        for (i, pp) in enumerate(pl.passes)
            t0 = time_ns()
            runpass!(pl, pp)
            # GPU first: on a backend that answers by submitting and waiting,
            # that wait is part of what this pass cost the frame, and the host
            # number is meant to include it.
            g = gpupasstime!(pl.graph.dev)
            sample!(prof.host_ns[i], Float64(time_ns() - t0))
            isnan(g) || sample!(prof.gpu_ns[i], g)
        end
    end
    # The frame is over: nothing may be left recorded and unsubmitted, whether
    # or not there is a window to present to.
    submit!(pl.graph.dev)
    for s in surfaces
        present_frame!(pl.graph.dev, s.win)
    end
    return nothing
end

"""
One pass: its updates, its dispatches, its draws and its copy, in that order.

The order is the graph's and it is the same on every backend. What each verb
DOES is the backend's — see `graphics/commands.jl`.
"""
function runpass!(pl::Plan, pp::PassPlan)
    # The update pass carries no dispatches: landing the writes an `Update`
    # ref is holding IS its body, and it happens HERE because this is the
    # graph position that reserved a `CopyDst` usage for them — before every
    # pass that declared a read.
    #
    # Nothing did this on a KernelAbstractions backend, so every `Update`
    # was silently dropped on both Host and Metal. Only the Vulkan route,
    # which records its passes itself, ever applied one.
    pp.pass.kind === :update && applyupdates!(writeupdate!, pl.graph.updates)
    if !isempty(pp.dispatches)
        # A backend may be holding recorded-but-unsubmitted drawing; a dispatch
        # must not overtake it. See `submit!`.
        submit!(pl.graph.dev)
        for s in pp.dispatches
            s()
        end
    end
    # A render pass runs after its own compute, in the order the graph put
    # them. The four verbs are the backend's; which attachments, which
    # draws and in what sequence is decided here — see
    # `graphics/commands.jl`.
    isempty(pp.draws) || runrenderpass!(pl, pp)
    # A copy pass has neither dispatches nor draws — its body is the one
    # transfer it declared, and nothing here ran it. Five of showcase's
    # twenty passes are copies, so on a backend without a batch queue the
    # whole deferred half of that frame read an empty g-buffer.
    pp.pass.kind === :copy && runcopypass!(pl, pp)
    return nothing
end

"""
    timings(plan) -> Vector{PassTiming}

What each pass cost, as a median over the last [`NSAMPLES`](@ref) frames.

`gpu_ms` is `NaN` unless the backend has timestamp queries — this reports what
`run!` itself can see, which is the host side. A backend with timestamps
overrides this and fills both.
"""
function timings(pl::Plan)
    prof = pl.profiler
    prof === nothing &&
        throw(ArgumentError("this plan was not built to be profiled; use Plan(g; profile = true)"))
    med(x) = isempty(x) ? NaN : (q = sort(x); q[(length(q) + 1) ÷ 2])
    return [PassTiming(prof.names[i], pl.passes[i].pass.kind,
                       med(prof.host_ns[i]) / 1e6, med(prof.gpu_ns[i]) / 1e6,
                       length(prof.host_ns[i])) for i in eachindex(prof.names)]
end

"""
    runcopypass!(plan, passplan)

Run a `copy!` pass: one target into one buffer.

Which resources, and that this happens between the pass that wrote the target
and the pass that reads the buffer, is the graph's; the transfer is the
backend's. The pair was checked at declaration, so there is nothing to validate
here.
"""
function runcopypass!(pl::Plan, pp::PassPlan)
    p = pp.pass
    copy_target!(pl.graph.dev, storage(p.dst), first_target(p))
    return nothing
end


"""
    runrenderpass!(plan, passplan)

Open the pass's attachments, record its draws in order, close it.

The sequencing is Mantle's and the four calls are the backend's. A backend that
records whole frames into a batch queue does not come through here — see
`supports_batch_queue`.
"""
function runrenderpass!(pl::Plan, pp::PassPlan)
    p = pp.pass
    dev = pl.graph.dev
    # `target_view`, not `storage`: an attachment is bound by whatever the
    # backend hands a render pass, and for a transient that is its view rather
    # than the arena it was placed in. Vulkan's own recording asks the same
    # question the same way.
    targets = Any[target_view(t) for t in p.targets]
    depth = p.depth === nothing ? nothing : target_view(p.depth)
    handle = begin_render_pass!(dev, targets, p.loads, depth, p.depth_load)
    try
        for d in pp.draws
            record_draw!(handle, d.compiled, d.args, d.count)
        end
    finally
        # Closed even if a draw throws: an encoder left open takes the whole
        # command buffer with it, and the next frame's failure would name the
        # wrong pass.
        end_render_pass!(handle)
    end
    return nothing
end
