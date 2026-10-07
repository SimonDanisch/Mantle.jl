# Running a compiled graph on a KernelAbstractions backend.
#
# Every backend whose kernels are KA kernels executes a plan the same way: bake
# one callable per dispatch at compile time, then a frame is a loop over
# callables. The Metal backend uses ALL of it — it overrides no function here, because `storage` already knows how to turn each
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
A TUPLE of graph arguments resolves element by element.

Without this a nested tuple reached `storage`, which has no method for one — so
an operand list could not be an argument, and a kernel over a variable number of
operands had to be written once per arity. `ew1!`/`ew2!`/`ew3!` in DNNKernels
were exactly that, and the arity limit was the API rather than the hardware.

Recursing through `resolve` rather than letting `storage(::Tuple)` do it, so a
backend that overrides `resolve` is consulted for the elements too. That is the
only difference between the two; `rawargs`, which has no device, goes through
`storage`.

Two other walks have to agree about tuples. `holdleaves!` walks them to their
leaves, so the resources stay held for the submission. `devicepointeroffsets`
must not count a tuple as a level of nesting (`nestinglevels` in
`graph/packing.jl`): stopping one short of the device pointers inside it gives a
tuple of operands an empty patch table, and a `resize!` under a recorded plan
then leaves the old address in place. The packer needs nothing: its generic
branch inlines any isbits aggregate and hands `recpatchfields!` the aggregate's
TYPE.

A tuple of plain values resolves to itself, since `storage(x) = x`.
"""
resolve(dev::Device, x::Tuple) = map(a -> resolve(dev, a), x)

"""
One launch, resolved as far as a launch can be.

`K`, `A` and `N` are concrete, so the call inside is direct: no lookup and no
splat over an abstract tuple. The one thing left for run time is the one that
means nothing otherwise — a device-computed count.
"""
struct Launch{K,A<:Tuple,N,G,D}
    kernel::K
    args::A
    ndrange::N
    # The workgroup size, or `nothing` when the kernel already carries one in its
    # type. See `callgroup`.
    group::G
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
finished.** On a GPU backend that is not free: the producing launch is queued, the host read races it, and the count comes back
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
of device memory races whatever wrote it.
"""
awaitwrites(dev::Device) = KernelAbstractions.synchronize(backend(dev))

# A count of zero launches nothing, which is what it means and what a recording
# backend does with it — an indirect dispatch of zero workgroups is a no-op
# there. Without this, KA gets an empty ndrange, and a wavefront round whose
# queue emptied is the ordinary case rather than an edge.
function (l::Launch{K,A,N,G})() where {K,A,N,G}
    nd = ndrangeof(l.device, l.ndrange)
    nd == 0 && return nothing
    if G === Nothing
        l.kernel(l.args...; ndrange = nd)
    else
        l.kernel(l.args...; ndrange = nd, workgroupsize = l.group)
    end
    return nothing
end

"""
    buildskernel(f, backend) -> Bool

Does `f` BUILD a kernel for `backend`, or is it one already?

`@kernel` generates a constructor: `f(backend)` returns the launchable object,
which is what `kernelfor` calls. A macro-free kernel is the function itself and
has no such method. The two cannot be told apart by `isa Function`, because the
macro generates a function too.

**`hasmethod(f, Tuple{typeof(backend)})` is NOT the question, though it was.** A
macro-free kernel that takes exactly one argument — `rngadvance_kernel!(state)`
— answers that yes, and the access walk then tried to construct it, reaching
`MethodError: no method matching rngadvance_kernel!(::LavaBackend, ::Int64)`
from inside `kernelfor`. What distinguishes them is what that call RETURNS:
`@kernel`'s constructor builds a `KernelAbstractions.Kernel`, and a macro-free
kernel called with a backend does not.

(It used to be asked by where the methods were defined — KernelAbstractions'
`macros.jl`. KernelAbstractions 0.10 relocates the generated constructors to the
`@kernel` line in the caller's file, so that test said "not a kernel" for every
`@kernel`, and the access walk tried to run the constructor as the kernel.)

One predicate, in core, because this is a decision about what a dispatch MEANS
and not about any backend's machinery: Lava's `compile_dispatch`, the ROCm
extension, `bake` below and `runlaunches!` all ask it here, and a caller that
does not ask is a `MethodError: no method matching ew2!(::CPU)` on the host.
Deletes itself with the last `@kernel`.
"""
function buildskernel(f, backend)
    hasmethod(f, Tuple{typeof(backend)}) || return false
    R = Base.infer_return_type(f, Tuple{typeof(backend)})
    return R !== Union{} && R <: KernelAbstractions.Kernel
end

"""
    kibackend(dev) -> backend

The backend object that compiles `KernelInterface` kernels for `dev`.

`backend(dev)` by default, and a verb of its own because it is not always the
same object: a device can speak both interfaces through two different types, as
ROCm does, where `backend(dev)` is the KernelAbstractions backend and the KI one
is a distinct type in another module. Which type `KI.kernel_function` is defined
on is the backend's own fact, so a caller asks for it rather than knowing it.

A backend with no KI implementation answers the KA one, and [`kikernel`](@ref)
then refuses a macro-free kernel there by name.
"""
kibackend(dev) = backend(dev)

"""
    kisupported(dev, f) -> Bool

Can `dev` compile the macro-free kernel `f` at all?
"""
kisupported(dev, f) =
    hasmethod(KI.kernel_function, Tuple{typeof(kibackend(dev)), typeof(f), Type})

"""
    kikernel(f, dev, args) -> KI.Kernel

The compiled kernel for a macro-free `f`, against the types its arguments will
arrive as.

`tt` is built from the ARGCONVERTED arguments because that is what the backend
passes at the launch, and a kernel compiled for the unconverted types is a
different kernel: a device array reaches the body as the backend's device-side
representation, not as the host wrapper.
"""
function kikernel(f, dev, args)
    be = kibackend(dev)
    kisupported(dev, f) || error(
        "Mantle: `dispatch!` was given `$f`, which is a macro-free kernel, and " *
        "$(typeof(be)) implements no `KI.kernel_function`. The backend answers " *
        "`KernelAbstractions` and not `KernelInterface`, so it can run an " *
        "`@kernel` but cannot compile a plain function.")
    tt = Base.to_tuple_type(map(a -> Core.Typeof(KI.argconvert(be, a)), args))
    return KI.kernel_function(be, f, tt)
end

"""Normalise KI's scalar / tuple / empty launch extents to a 3-tuple."""
@inline function ki_extent(x, default::Int)
    x isa Integer && return (Int(x), default, default)
    n = length(x)
    n == 0 && return (default, default, default)
    return (Int(x[1]),
            n >= 2 ? Int(x[2]) : default,
            n >= 3 ? Int(x[3]) : default)
end

"""
    ki_launch_extents(backend, ndrange, workgroupsize, numworkgroups;
                      max_work_group_size) -> (wg, blocks)

The workgroup size and workgroup count a KI launch resolves to, as 3-tuples.

Shared by every backend's immediate KI launch (Vulkan's, the host's) and by
Vulkan's `compile_dispatch`, which needs the same answer at COMPILE time — a
recorded plan does no host work per run, so the extents have to be fixed when the
plan is built. One copy, because two would be one place for the two paths to
disagree about the same launch.

`threads_to_workgroupsize` is KI's, and it SHAPES the device limit to the ndrange
rather than putting it all on the first axis. Getting that wrong by hand cost 29%
of SAM 2.1's encoder on the ROCm side; here it is upstream's arithmetic.
"""
function ki_launch_extents(backend, ndrange, workgroupsize, numworkgroups;
                           max_work_group_size::Int = typemax(Int))
    limit = min(max_work_group_size, KI.max_work_group_size(backend))
    if length(ndrange) > 0
        nd = ki_extent(ndrange, 1)
        wg = length(workgroupsize) > 0 ? ki_extent(workgroupsize, 1) :
             ki_extent(KI.threads_to_workgroupsize(limit, nd), 1)
        blocks = ntuple(i -> cld(nd[i], wg[i]), 3)
    else
        # KI's default is one workgroup of one workitem.
        wg     = ki_extent(workgroupsize, 1)
        blocks = ki_extent(numworkgroups, 1)
    end
    prod(wg) <= limit ||
        throw(ArgumentError("workgroupsize $wg exceeds the device limit of $limit workitems"))
    return wg, blocks
end

kernelfor(k, ::Nothing, backend) = k(backend)
kernelfor(k, group, backend) = k(backend, group)

"""
    callgroup(kernel, group) -> group or nothing

The workgroup size a launch still has to pass at the CALL, having built `kernel`
with [`kernelfor`](@ref).

`nothing` for a kernel that carries its workgroup in its TYPE: KA reads it from
there, and passing it again keys a second, identical iteration plan for the same
launch. That was the whole rule, and it is wrong for the other kind. A kernel
whose workgroup is `DynamicSize` was launched as
`k(backend)(args...; ndrange, workgroupsize = wg)` — the size is in the call and
nowhere else, and an intercepted launch that drops it gets KA's default instead.

Silently, and only for kernels that also read their own workgroup size from a
`Val` argument, which the broadcast kernels here do: a body compiled for
`Val(256)` dispatched in groups of 64 writes `base = (g-1)*256 + l` for `l` in
`1:64`, so **one element in four** is written and the rest keep whatever was in
the buffer. The rest of the graph replays bit-exact, which is what made it look
like a hazard.
"""
callgroup(::KernelAbstractions.Kernel{<:Any,<:KernelAbstractions.NDIteration.StaticSize},
          group) = nothing
callgroup(::KernelAbstractions.Kernel, group) = group

# A `Launch` resolved its count at `bake`: a `DeviceRange` with a ceiling became
# the ceiling, one without became a `DeferredRange`. Neither is sized on the
# device the way a recording backend means it — see `PassPlan.indirect`.
devicesized(::Launch) = false

# A function barrier: `args` is concrete inside `Launch`, which is what makes
# the call in `(l::Launch)()` specialise.
"""
What a CALL compiles to: the function and its resolved arguments.

Nothing else. A call has no kernel to build, no launch geometry to work out and
no argument memory to lay out — the library on the other side does all three —
so what the plan holds is the call itself, which is why `argsize` and
`indirectindex` are zero and `devicesized` is false.

Core's, and one type rather than one per backend: the ROCm extension had a
`ROCmCallDispatch` that was this with an `ndrange` keyword bolted on for the old
callable-kernel convention, and it became unreachable the moment
[`buildskernel`](@ref) started routing every non-`@kernel` to the
KernelInterface path.
"""
struct Call{F,A<:Tuple,D}
    f::F
    # The DECLARED arguments, not resolved ones. A plan bakes its launches once,
    # but an arena is allowed to MOVE afterwards: a later, larger plan placed in
    # the same arena grows it, `remap!` re-materialises every transient into the
    # new region and `notify_move!` shifts every baked device address. A launch
    # follows because its addresses live in argument memory the patch table
    # covers; a call has no argument memory, so a resolved operand held here
    # would be a device array over the region the arena USED to have. `record!`
    # already refuses a kernel that closes over a device array, for this reason
    # and in those words — freezing the operands here was the same mistake one
    # level down.
    #
    # Nothing is lost by deferring it. `resolve` is `storage`, a field read, and
    # the library on the other side of a call costs orders more than reading a
    # pointer out of each operand.
    args::A
    dev::D
end

(c::Call)() = (c.f(resolve(c.dev, c.args)...); nothing)
argsize(::Call) = 0
indirectindex(::Call) = 0
devicesized(::Call) = false

function bake(c::Compile, d::Dispatch)
    dev = c.graph.dev
    be = backend(dev)
    args = map(a -> resolve(dev, a), d.args)
    # A call is compiled by resolving its arguments and nothing else — and
    # refused here, once, rather than by each backend, because whether a call
    # can run is one question with one answer per backend.
    if iscall(d)
        runscalls(dev) || throw(ArgumentError(
            "dispatch!: `$(d.kernel)` was declared as a call (no ndrange), and " *
            "$(typeof(dev)) cannot run one: its `run!` submits a recording it " *
            "built as a command buffer, and a host call cannot be written into " *
            "one. Declare the operation as dispatches instead — on this backend " *
            "a GEMM is `coopmat_gemm!`, which records like any other dispatch."))
        return Call(d.kernel, d.args, dev)
    end
    # A `KI.Kernel` is callable with the same `ndrange`/`workgroupsize` keywords
    # a `KA.Kernel` is (KI's `Kernel` docstring states that contract), so the
    # only difference here is which object gets built, and `(l::Launch)()` needs
    # to know nothing about it.
    buildskernel(d.kernel, be) || return Launch(kikernel(d.kernel, dev, args), args,
                                                bakedrange(c, d.ndrange), d.group, dev)
    let k = kernelfor(d.kernel, d.group, be)
        Launch(k, args, bakedrange(c, d.ndrange), callgroup(k, d.group), dev)
    end
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
# A `DeferredRange` here instead calls `awaitwrites` from `ndrangeof`: a full
# device synchronise, on the host, before EVERY such dispatch. Measured on an
# M5, killeroo-gold at 684x513/8spp, 320 of them per frame cost 0.585 s of a
# 0.602 s frame, and hardware traversal made no difference because traversal
# was not the cost.
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
and the size of the block it will pack.
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
"""
The rectangle a draw is clipped to, resolved once at compile rather than left
for the emitter to fall back on.

Resolved HERE because a draw has to be self-contained: an emitter that sets a
viewport only when the draw names one leaks the previous draw's rectangle into
the next draw that names none, and which draw came before depends on an
ordering the scheduler is free to change. After this, `nothing` means nothing in
the pass ever asked, which is the one case where whatever the backend opened the
pass with is the right answer.

That leaves one shape that cannot be resolved and is therefore rejected: some
draws naming a rectangle, others not, and no default on the pass. The extent the
others want is the whole target, but its SIGN is a convention the caller owns —
a backend that mirrors y wants a negative height to undo the mirror — so core
declines to guess and asks for `render!(...; viewport = ...)`.
"""
function drawviewport(p::Pass, d)
    d.viewport === nothing || return d.viewport
    p.viewport === nothing || return p.viewport
    any(o -> o.viewport !== nothing, p.draws) && throw(ArgumentError(
        "pass \"$(p.name)\": some draws set a viewport and this one does not, " *
        "so its rectangle would be whichever draw happened to run before it. " *
        "Give the pass a default with render!(...; viewport = (x, y, w, h))."))
    return nothing
end

"""
Which of a draw's two argument lists gets packed — a pipeline has one push
constant range, so exactly one of them does.

A [`DrawBinding`](@ref) always wins, and that is not a preference: it is the
only one of the three that a `rebind!` can reach. Picking the resolved
`frag_args` tuple instead left the cell compiled-in but never read, so a plan
was correctly reused across a zoom and correctly drew the same picture every
time — the failure that looks most like success.

Otherwise it is whichever list is non-empty, and when both are the same list
(every Makie render object) either answers.
"""
@inline packedargs(b::DrawBinding, _, _) = b
@inline packedargs(::Tuple, vargs, fargs) = isempty(fargs) ? vargs : fargs

function compiledraw(c::Compile, p::Pass, d, argoff::Int)
    dev = c.graph.dev
    color_formats = [attachment_format(t) for t in p.targets]
    depth_format  = p.depth === nothing ? nothing : attachment_format(p.depth)
    # `drawargs` first: a rebindable cell is already resolved, and `resolve` on a
    # resolved argument is the identity, so this is the same tuple either way.
    # What the COMPILE sees is the cell's contents at this moment, which is right
    # — its types are what the pipeline is built around, and `rebind!` cannot
    # change them.
    vargs = map(a -> resolve(dev, a), drawargs(d.args))
    fargs = map(a -> resolve(dev, a), d.frag_args)
    compiled = compile_draw(dev, d.shader, color_formats, depth_format, vargs, fargs;
                            bindings = d.bindings)
    indices = d.indices === nothing ? nothing : resolve(dev, d.indices)
    # ONE list, the way the Vulkan front picks `packed`. Concatenating was the
    # same thing while one of the two was always empty; with both stages allowed
    # the same list it would pack every argument twice.
    stageargs = packedargs(d.args, vargs, fargs)
    return CompiledDraw(compiled, d.shader, stageargs, bakedcount(c, d.count),
                        drawviewport(p, d), d.bindings, indices, d.instances, argoff, 0)
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
# A transient's storage is the arena slice the placer gave it, which on a
# KernelAbstractions backend is an array over the slab — so the write is the
# copy, in place, with no staging to route it through.
landhost!(t::TransientBuffer, data::AbstractVector, from::Integer) =
    (copyto!(view(storage(t), from:(from + length(data) - 1)), data); nothing)

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
        # there, and the framebuffer size is asked again), and refitting before
        # it fits against a swapchain the acquire then rebuilds: a colour target
        # at the new size beside a depth target at the old one, which is a lost
        # device on NVIDIA. A frame where nothing moved costs one
        # size compare per tracking transient, and `refit!` returns `false` for
        # a plan that follows nothing.
        refit!(pl)
        checkextents(pl)
        # A plan measuring itself takes the last run's pass times here, and
        # re-cuts its recording (below) once they are all in.
        measure!(pl)
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
    # Before this run overwrites them, and without waiting: see `collect!`. Only
    # a recording that writes timestamps has any to read; a measured plan keeps
    # its profiler and must not pay a dynamic call for it every run.
    pl.stamped && collect!(pl.profiler, dev)
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
one is walked into the run, pass by pass: an interpreted backend always, and a
windowed plan on any backend.

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
            emitplan!(e, pl)
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
    pl.stamped && (pl.profiler.pending = true)
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
        isempty(p) ? nothing : (pl.pending_patches = Tuple{Any,Int,UInt64}[]; p)
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
(s::StoreEmitter)(b::Union{Buffer,TransientBuffer}, data::AbstractVector, from::Integer) =
    emitstore!(s.e, b, data, from)

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
emitstore!(e, b::Buffer{T}, data::AbstractVector{T}, from::Integer) where {T} =
    storefrom!(e, b.store, data, from)
# A transient names ITSELF as the target and not `storage(t)`: the backend needs
# the block to name the buffer a copy writes into, and an array view over it has
# dropped that (see the note on the struct's `block`).
emitstore!(e, t::TransientBuffer{T}, data::AbstractVector{T}, from::Integer) where {T} =
    storefrom!(e, t, data, from)

"""The bytes of one store, at an element offset into whatever the backend was
handed. Shared by both `emitstore!` methods, which differ only in the target."""
function storefrom!(e, target, data::AbstractVector{T}, from::Integer) where {T}
    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src storebytes!(e, target, (Int(from) - 1) * sizeof(T),
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
    # The profiler a plan measures its budget with is not a request for timings:
    # its samples are the first run's and stop when the measurement ends.
    pl.profile ||
        throw(ArgumentError("this plan was not built to be profiled; use Plan(g; profile = true)"))
    prof = pl.profiler
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
# the spot. The Vulkan backend carried a second copy of this walk — `emitplan!`,
# `emitpass!`, `emitwork!` — with `record!` hung off it, which put the
# sequence in a backend and left core with a stub.

"""
    emitplan!(emitter, plan)

Walk the plan's passes into `emitter`: the head barrier, then every pass —
its barriers, its predicate scope and its work — in compile order.

Named for what it walks, beside `emithead!`, `emitpass!` and `emitwork!`.
`emit!` is a geometry body emitting a vertex, which is a different thing.
"""
function emitplan!(e, pl::Plan)
    emithead!(e, pl)
    # A loop body is walked once per iteration, behind its loop head. Only a
    # WALKED plan gets here with a loop (an interpreted run, a windowed frame):
    # a recording cuts a loop into pieces of its own (`partitionranges`) and
    # repeats them at submission.
    i = 1
    for (r, loop) in pl.loops
        emitpasses!(e, pl, i:(first(r) - 1))
        for _ in 1:loop.count
            emitloophead!(e, pl, r)
            emitpasses!(e, pl, r)
        end
        i = last(r) + 1
    end
    emitpasses!(e, pl, i:length(pl.passes))
    return nothing
end

"""
The passes at `range`, with the profiler's timestamps around each if the plan has
one.

Separate from [`emitplan!`](@ref) because a PARTITIONED recording emits a slice
into each of its pieces and the head into the first only — see
[`recordparts!`](@ref). One walk, sliced; not a second walk.
"""
function emitpasses!(e, pl::Plan, range)
    # Two loops rather than one with a branch in it: a frame that is not being
    # profiled must not pay a comparison per pass.
    if !pl.stamped
        for i in range
            emitpass!(e, pl, pl.passes[i])
        end
    else
        for i in range
            profiled!(pl, e, i) do
                emitpass!(e, pl, pl.passes[i])
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
    # groups for a discarded one. Inside the scope it would be discarded with
    # the iteration, and a trace-rays command is not subject to conditional
    # rendering (VK_KHR_ray_tracing_pipeline leaves it out), so the trace of a
    # discarded bounce would still run, with a ray count read from a slot
    # nothing had written. `test_discarded_iteration_prepare.jl` pins the slot.
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

It does NOT run the plan.

Where the recording is cut into submissions is core's, and no caller chooses it:
see [`partitionranges`](@ref). A plan that has not been measured yet is cut after
every pass and timed; `run!` re-records it once from what it measured, into
submissions of at most the plan's `budget` of GPU time.
"""
function record!(pl::Plan)
    pl.recording === nothing || return pl
    recordable(pl) || throw(ArgumentError(
        isempty(pl.graph.surfaces) ?
        "record!: this plan holds a rebindable draw binding. A recording packs a " *
        "draw's arguments once and holds their address for its life, so a later " *
        "`rebind!` would be written to memory nothing reads. Leave the plan " *
        "unrecorded — it re-packs every run, which is what a cell is for." :
        "record!: this plan draws to a surface. A swapchain image is a different " *
        "image every frame and a recording names one, so a windowed plan needs a " *
        "recording per swapchain image — which is not built yet. Headless plans " *
        "record today."))
    recordplan!(pl.graph.dev, pl)
    return pl
end

function recordplan!(dev, pl::Plan)
    # The pieces this plan records as: one range covering everything for a
    # measured plan that fits its budget and has no `when!` condition, which is
    # the single-piece path below.
    ranges = partitionranges(pl)
    # The measuring recording carries the timestamps; a measured one only if
    # the caller asked for timings.
    pl.stamped = pl.profiler !== nothing && (pl.profile || measuring(pl))
    e = openrecording(dev, pl, first(ranges))
    if e === nothing
        # A backend with no command buffers to build: `run!` walks the plan
        # instead, which is the same walk, and there are no submissions to cut.
        # A `when!` region is refused: skipping one means not submitting the
        # piece it is in, and a backend with no pieces has nothing to skip.
        any(pp -> pp.pass.hostcond !== nothing, pl.passes) && throw(ArgumentError(
            "record!: this plan has a `when!` condition and this backend does " *
            "not record: a conditional region is a recorded piece that the run " *
            "may omit, and `openrecording` declined the plan. Branch on the " *
            "host and declare the graph you want."))
        pl.stamped = pl.profile
        return pl
    end
    # Where every device address the pack writes lands, keyed by the address —
    # derived once, from this one pack, by `closerecording!`; consulted only by
    # `notify_move!`, never per run. A partition fills it from every piece.
    empty!(pl.patchtab)
    empty!(pl.pending_patches)
    if length(ranges) == 1
        emitplan!(e, pl)
        pl.recording = closerecording!(e, pl)
    else
        pl.recording = recordparts!(dev, pl, e, ranges)
    end
    pl.partition = ranges
    listen_moves!(pool(pl.graph.dev), pl)
    return pl
end

"""
    recordparts!(device, plan, emitter, ranges) -> RecordingParts

The chunked walk: each of [`partitionranges`](@ref)' ranges into a piece of its
own, `emithead!` into the first only, and every piece kept reachable so a throw
part way through releases all of them rather than leaking a command buffer per
chunk.

`emitter` is the already-open first piece, because `recordplan!` has to open one
to find out whether this backend records at all.

Nothing here is a driver call, which is why it is core's: the arithmetic that
picks the chunks is the same everywhere, and a piece is whatever this backend's
`closerecording!` hands back.
"""
function recordparts!(dev, pl::Plan, emitter, ranges)
    parts = Any[]
    conds = Any[]
    e = emitter
    heads = Set(first(r) for (r, _) in pl.loops)
    try
        for chunk in ranges
            e === nothing && (e = openrecording(dev, pl, chunk))
            first(chunk) == 1 && emithead!(e, pl)
            # The first piece of a loop body opens every iteration: a full
            # barrier, so what the last iteration wrote is visible to this one.
            isempty(chunk) || first(chunk) in heads &&
                emitloophead!(e, pl, only(r for (r, _) in pl.loops if first(r) == first(chunk)))
            emitpasses!(e, pl, chunk)
            push!(parts, closerecording!(e, pl))
            push!(conds, isempty(chunk) ? nothing : pl.passes[first(chunk)].pass.hostcond)
            e = nothing                 # closed, and now `parts`' business
        end
    catch
        # The open one is not in `parts` and cannot be released as a recording:
        # it was never closed.
        e === nothing || abandonrecording!(dev, e)
        foreach(release!, parts)
        empty!(pl.patchtab)
        rethrow()
    end
    # Narrowed from `Any[]`: the pieces are all one backend type, and
    # `submitrecording!` over them should specialise.
    parts = identity.(parts)
    batches, batchconds = submissionbatches(pl, ranges, parts, conds)
    return RecordingParts(parts, conds, batches, batchconds)
end

"""
    submissionbatches(plan, ranges, parts, conds) -> (batches, conds)

The pieces in the order a run submits them, grouped into submissions.

The ORDER repeats a loop body's pieces once per iteration. The GROUPING puts
consecutive pieces into one submission while their measured cost fits the plan's
budget, so a loop whose iterations are cheap is still one submission and one that
is not is cut between iterations; a piece not measured yet is a submission of its
own. A conditional (`when!`) piece is always its own, so a run can leave it out.
"""
function submissionbatches(pl::Plan, ranges, parts::Vector{R}, conds) where {R}
    loopof(r) = isempty(r) ? nothing : pl.passes[first(r)].pass.loop
    order = Int[]
    k = 1
    while k <= length(ranges)
        l = loopof(ranges[k])
        if l === nothing
            push!(order, k)
            k += 1
        else
            j = k
            while j < length(ranges) && loopof(ranges[j + 1]) === l
                j += 1
            end
            for _ in 1:l.count
                append!(order, k:j)
            end
            k = j + 1
        end
    end
    timed = measurable(pl)
    cost(k) = timed ? sum(i -> isnan(pl.passcost[i]) ? Inf : pl.passcost[i], ranges[k]; init = 0.0) : 0.0
    batches = Vector{R}[]
    bconds = Any[]
    t = Inf
    for k in order
        c = cost(k)
        if conds[k] !== nothing || !isempty(bconds) && bconds[end] !== nothing ||
           isempty(batches) || t + c > pl.budget
            push!(batches, R[])
            push!(bconds, conds[k])
            t = 0.0
        end
        push!(batches[end], parts[k])
        t += c
    end
    return batches, bconds
end

"""
    partitionranges(plan) -> Vector{UnitRange{Int}}

The pass ranges each recorded piece covers, and so each submission.

Two things cut a piece, and they are independent:

  * the plan's `budget` of GPU time per submission, against what each pass was
    measured to cost. A pass not measured yet is taken to be unbounded, so an
    unmeasured plan is cut after every pass: nothing else bounds how long a pass
    runs, and a submission the watchdog kills returns garbage with no error. A
    device without timestamps cannot measure and keeps one piece, and so does
    `budget = Inf`;
  * a change of [`when!`](@ref) condition, because a piece is the unit that can
    be left unsubmitted, so a conditional region has to be one;
  * the edge of a [`repeat!`](@ref) body, because a run submits the body's pieces
    once per iteration.

The FIRST range is always unconditional, and that is not a convenience: the head
(`emithead!`) is baked into piece one and carries the head barrier and the
profiler's query-pool reset. A piece that might not be submitted cannot hold it,
so a graph whose very first pass is conditional gets an empty leading piece
rather than a head that sometimes does not run.
"""
function partitionranges(pl::Plan)
    n = length(pl.passes)
    out = UnitRange{Int}[]
    n == 0 && return push!(out, 1:0)
    cond(i) = pl.passes[i].pass.hostcond
    loop(i) = pl.passes[i].pass.loop
    budget = pl.budget
    timed = measurable(pl)
    # Untimed is unbounded: a submission of its own at any finite budget, and
    # `budget = Inf` is the one that keeps everything together regardless.
    cost(i) = timed ? (isnan(pl.passcost[i]) ? Inf : pl.passcost[i]) : 0.0
    # An empty leading piece when the graph opens on a conditional region or a
    # loop: the head has to go somewhere that runs once, always.
    (cond(1) === nothing && loop(1) === nothing) || push!(out, 1:0)
    i = 1
    while i <= n
        c = cond(i)
        l = loop(i)
        j = i
        t = cost(i)
        # Extend while the condition and the loop are the SAME OBJECTS and the
        # budget allows: a piece never straddles a loop body's edge, because the
        # body's pieces are the ones a run submits once per iteration.
        while j < n && cond(j + 1) === c && loop(j + 1) === l && t + cost(j + 1) <= budget
            j += 1
            t += cost(j)
        end
        push!(out, i:j)
        i = j + 1
    end
    return out
end

"""Whether this plan can time its passes, and so cut itself by its budget. An
infinite budget never cuts, so there is nothing to measure for."""
measurable(pl::Plan) =
    pl.profiler !== nothing && !isnan(pl.profiler.period_ns) && isfinite(pl.budget)

"""Whether this plan still has an unconditional pass it has not timed."""
function measuring(pl::Plan)
    measurable(pl) || return false
    for i in eachindex(pl.passcost)
        isnan(pl.passcost[i]) && pl.passes[i].pass.hostcond === nothing && return true
    end
    return false
end

"""
    measure!(plan) -> plan

Take the pass times the last timed run left into `passcost`, and throw the
recording away when they change how it is cut — or when they complete the
measurement and the timestamps it carried can go. `run!` then records again
before it submits, which happens once in a plan's life.

The WORST sample is what counts: a budget is a bound, and the first run of a
pass is often its slowest.

A `repeat!` body writes its timestamps again every iteration and what is read is
the last iteration's. So a gated loop whose last iteration the gate discarded is
measured as nearly free, and its iterations are batched as if they were: a
budget that holds for plain loops and is only as good as that last iteration for
a gated one.
"""
function measure!(pl::Plan)
    # Only a recording that carries timestamps has anything to measure, and a
    # measured plan's does not: its runs pay one field read here.
    (pl.stamped && measuring(pl)) || return pl
    prof = pl.profiler
    for i in eachindex(pl.passcost)
        isnan(pl.passcost[i]) || continue
        s = prof.gpu_ns[i]
        isempty(s) || (pl.passcost[i] = maximum(s) * 1e-9)
    end
    # Only once every unconditional pass is timed: re-cutting on a partial
    # measurement re-records again when the rest arrives.
    measuring(pl) && return pl
    pl.recording === nothing && return pl
    (partitionranges(pl) != pl.partition || (pl.stamped && !pl.profile)) && invalidate!(pl)
    return pl
end

"""
The baked pieces of one plan, in the order they are submitted.

A recording is cut into several pieces (see [`partitionranges`](@ref)) so that no
submission runs longer than the plan's budget, and so a workload that would
exceed the driver's submission timeout has completion points inside it. They remain ONE plan: the
barriers between the pieces are the ones the graph derived, and a piece is
submitted only after the one before it has been.

The type is core's and so is the partition that builds it — see
[`recordparts!`](@ref). A backend answers `submitrecording!` for ONE piece and
gets the sequence for free.
"""
struct RecordingParts{R}
    parts::Vector{R}
    # One per piece: `nothing` for a piece that always runs, or the [`when!`](@ref)
    # condition the host asks at submit time. Parallel to `parts` rather than a
    # field on the piece, because a piece is whatever the backend's
    # `closerecording!` hands back and core does not get to add to it.
    conds::Vector{Any}
    # What a run submits, in order: the pieces grouped into submissions, a loop
    # body's repeated once per iteration (`submissionbatches`). Each is one
    # submission, handed to the backend as one vector so a backend that can put
    # several command buffers in one submission does.
    batches::Vector{Vector{R}}
    # One per batch: `nothing`, or the condition of the one conditional piece
    # the batch is.
    batchconds::Vector{Any}
end

release!(rec::RecordingParts) = (foreach(release!, rec.parts); nothing)

"""
    submitrecording!(ctx, recording, emitter) -> token

Submit one baked piece, or in this method the sequence a partition made of one.

The iteration is core's; what a submission IS belongs to the backend, which
answers the same name for its own recording type. The run's own emitter rides
with the FIRST piece — it carries this run's host stores and address patches,
which have to land before any baked command reads them — and the rest go alone.
"""
function submitrecording!(ctx, rec::RecordingParts, e)
    tok = nothing
    pending = e
    for k in eachindex(rec.batches)
        c = rec.batchconds[k]
        c === nothing || c[] || continue
        tok = submitrecording!(ctx, rec.batches[k], pending)
        pending = nothing
    end
    # Every batch was conditional and every condition false. The run still has to
    # land its host stores and patches, and something has to answer for the
    # submission, so the unconditional head goes on its own.
    pending === nothing || (tok = submitrecording!(ctx, first(rec.parts), pending))
    return tok
end

"""
    submitplan!(ctx, plan, emitter, R) -> token

Submit what `plan` recorded, `R` being this backend's piece type.

`Plan.recording` is an `Any` field, so a call through it is dynamic, and the
`UInt64` token a dynamic call returns comes back boxed: 8 bytes a run once the
timeline passes the small-integer cache. Narrowed here to the three things a
recording can be — one piece, the pieces of a partition, or `nothing` for a plan
that was not recorded — so every call below is static. Vulkan and Metal submit
through this rather than reading the field themselves; Metal did read it, and its
runs allocated the box (RayMakie `test_render_allocates_nothing.jl`). The ROCm
extension still reads the field, and boxes nothing doing so: its
`submitrecording!` returns `nothing`, and its token comes from `fence`.
"""
function submitplan!(ctx, pl::Plan, e, ::Type{R}) where {R}
    rec = pl.recording
    rec isa R && return submitrecording!(ctx, rec, e)
    rec === nothing && return submitrecording!(ctx, nothing, e)
    return submitrecording!(ctx, rec::RecordingParts{R}, e)
end

"""
    submitrecording!(ctx, pieces::AbstractVector, emitter) -> token

One submission made of several pieces, in order. A backend that can put several
command buffers in one submission answers this for its own piece type; this
default submits them one after another, which is the same order and a completion
point between each.
"""
function submitrecording!(ctx, pieces::AbstractVector, e)
    tok = submitrecording!(ctx, pieces[1], e)
    for k in 2:length(pieces)
        tok = submitrecording!(ctx, pieces[k], nothing)
    end
    return tok
end

"""
Whether this plan's commands can be written once, or have to be emitted per run.

Two reasons they cannot, and both are the same shape — something inside the
commands differs between runs:

  * a plan drawing to a SURFACE names a swapchain image, and that is a different
    image every frame;
  * a plan holding a [`DrawBinding`](@ref) names the bytes it packed, and a cell
    whose whole purpose is to be rewritten between runs cannot be frozen into
    them.

Refused here and not in the packer, which cannot tell whether it is packing a
recording or a run: this plan cannot be recorded, so it is emitted per run, and
the packer re-reads every cell on the way. A Makie frame, whose every draw is a
cell, depends on that.
"""
recordable(pl::Plan) = isempty(pl.graph.surfaces) && !hasrebindable(pl)

"""Does any draw in this plan take its arguments from a cell the host rewrites?"""
function hasrebindable(pl::Plan)
    for p in pl.graph.passes, d in p.draws
        d.args isa DrawBinding && return true
    end
    return false
end

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

# Walked on the host, one pass after another: nothing is in flight to order.
emitloophead!(::Immediate, ::Plan, ::AbstractUnitRange) = nothing

function emitcopy!(::Immediate, pl::Plan, pp::PassPlan)
    p = pp.pass
    if p.viewport === nothing
        copy_target!(pl.graph.dev, storage(p.dst), first_target(p))
    else
        copy_target!(pl.graph.dev, storage(p.dst), first_target(p); region = p.viewport)
    end
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
function emitdraw!(::Immediate, handle, d)
    d.viewport === nothing || setviewport!(handle, d.viewport...)
    # Through the binding: a texture table is a per-frame value like the rest.
    bind = boundbindings(d.args, d.bindings)
    bind === nothing || use_bindings!(handle, d.compiled, bind)
    # Every one of these through the binding when there is one: a tick count is a
    # vertex count and a relaid-out plot is a new index buffer, so freezing them
    # would rebuild a plan for a zoom just as surely as freezing the camera did.
    record_draw!(handle, d.compiled, drawargs(d.args), boundcount(d.args, d.count);
                 instances = boundinstances(d.args, d.instances),
                 indices = boundindices(d.args, d.indices))
    return nothing
end
endrender!(::Immediate, handle) = (end_render_pass!(handle); nothing)

"""
A pointer patch into the plan's own argument memory, written by the host.

The `Immediate` front is what a backend opens when its runs are already ordered
against each other before core gets here — Metal drains the device in `openrun`
when the plan has a recording — so the eight bytes can simply be stored. A
backend that pipelines instead orders the patch by putting it IN the submission,
which is what `emitinline!` on a real command emitter does.

`ArgMemory` carries its own host pointer, so nothing is asked of the backend at
all: unified memory or not, a plan's argument memory is host-written — that is
how it was packed in the first place.
"""
function emitinline!(::Immediate, am::ArgMemory, off::Int, p::Ptr{Cvoid}, n::Int)
    unsafe_copyto!(am.ptr + off, convert(Ptr{UInt8}, p), n)
    return nothing
end
