# Running a compiled graph: baking dispatches into callables (`Launch`, `Call`),
# the `Pipelines` phase, `run!`/`execute!`, the pass walk, recording and its
# partition into submissions, and the `Immediate` emitter. Backends specialise
# `compiledraw`/`compile_dispatch` on their `Compile{D}` and implement the emit
# primitives declared in `graph/backend.jl`.

"""
    resolve(device, x)

The value a baked launch passes for graph argument `x`: [`storage`](@ref) by
default.
"""
resolve(::Device, x) = storage(x)

"""
    resolve(device, x::Tuple)

Resolve a tuple argument element by element through `resolve`, so a backend
override applies to the elements. `holdleaves!` and `nestinglevels` (which must
not count a tuple as a nesting level, or its pointers go unpatched on `resize!`)
treat tuples the same way.
"""
resolve(dev::Device, x::Tuple) = argmap(a -> resolve(dev, a), x)

"""
    Launch(kernel, args, ndrange, group, device)

A baked KA or KI launch. The concrete type parameters make `(l::Launch)()` a
static call; only a `DeferredRange` count is read at run time.
"""
struct Launch{K,A<:Tuple,N,G,D}
    kernel::K
    args::A
    ndrange::N
    # `nothing` when the kernel carries its workgroup size in its type (`callgroup`).
    group::G
    # For `ndrangeof` to wait on before reading a deferred count.
    device::D
end

"""
    DeferredRange(store)

A device-written dispatch count read at launch, not at compile, because an
earlier pass writes it.
"""
struct DeferredRange{S}
    store::S
end

ndrangeof(::Device, n) = n

"""
    ndrangeof(device, r::DeferredRange) -> Int

Read a device-written count on the host at launch. Calls
[`awaitwrites`](@ref) first: without it the read races the producing kernel and
a stale count silently launches too few threads (seen as flickering pixels in
Hikari's wavefront queues on Apple M5).
"""
function ndrangeof(dev::Device, r::DeferredRange)
    awaitwrites(dev)
    return Int(first(r.store))
end

"""
    awaitwrites(device)

Wait for submitted device work before the host reads memory it wrote.
Defaults to `KernelAbstractions.synchronize`.
"""
awaitwrites(dev::Device) = KernelAbstractions.synchronize(backend(dev))

"""
Eager `kernel(args...; ndrange = DeviceRange(count; max))` on a KA backend
without indirect dispatch (Vulkan handles it in its own launch). As in
`bakedrange`: with `max` the grid is `max` and the kernel's bounds check covers
the surplus; without it the count is copied to the host.
"""
function KernelAbstractions.launch_config(kernel::KernelAbstractions.Kernel, r::DeviceRange,
                                          workgroupsize)
    n = r.max === nothing ? Int(first(Array(storage(r.count)))) : r.max
    return KernelAbstractions.launch_config(kernel, (n,), workgroupsize)
end

# A zero count launches nothing, like an indirect dispatch of zero groups; KA
# would fail on an empty ndrange.
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

Whether `f` is a `@kernel` constructor, i.e. `f(backend)` returns a
`KernelAbstractions.Kernel`, rather than a macro-free kernel. Decided by the
inferred return type: `hasmethod` alone misreads a one-argument macro-free
kernel, and KernelAbstractions 0.10 defines constructors in the caller's file.
"""
function buildskernel(f, backend)
    hasmethod(f, Tuple{typeof(backend)}) || return false
    R = Base.infer_return_type(f, Tuple{typeof(backend)})
    return R !== Union{} && R <: KernelAbstractions.Kernel
end

"""
    kibackend(dev) -> backend

The backend object that compiles `KernelInterface` kernels for `dev`:
`backend(dev)` unless the device uses a separate KI type (ROCm). Without KI
support [`kikernel`](@ref) refuses macro-free kernels.
"""
kibackend(dev) = backend(dev)

"""
    kisupported(dev, f) -> Bool

Whether `dev` can compile the macro-free kernel `f`.
"""
function kisupported(dev, f)
    be = kibackend(dev)
    applicable(KI.argconvert, be, f) || return false
    converted = KI.argconvert(be, f)
    return hasmethod(KI.kernel_function, Tuple{typeof(be), typeof(converted), Type})
end

"""
    kikernel(f, dev, args) -> KI.Kernel

The compiled kernel for a macro-free `f`, specialised on the `KI.argconvert`ed
argument types, which is what the backend passes at launch.
"""
function kikernel(f, dev, args)
    be = kibackend(dev)
    kisupported(dev, f) || error(
        "Mantle: `dispatch!` was given `$f`, which is a macro-free kernel, and " *
        "$(typeof(be)) implements no `KI.kernel_function`. The backend answers " *
        "`KernelAbstractions` and not `KernelInterface`, so it can run an " *
        "`@kernel` but cannot compile a plain function.")
    tt = Base.to_tuple_type(map(a -> Core.Typeof(KI.argconvert(be, a)), args))
    return KI.kernel_function(be, KI.argconvert(be, f), tt)
end

"""
    ki_launch_extents(kernel::KI.Kernel, ndrange, workgroupsize, numgroups;
                      max_work_group_size) -> (wg, blocks)

The workgroup size and count a KI launch resolves to, as 3-tuples, from KI's
own `launch_geometry`, so recording backends that fix extents at compile time
agree with `KI.launch`. It spreads the device limit across the ndrange's axes.
"""
function ki_launch_extents(kernel::KI.Kernel, ndrange, workgroupsize, numgroups;
                           max_work_group_size::Int = typemax(Int))
    blocks, wg = KI.launch_geometry(kernel, numgroups, workgroupsize, ndrange,
                                    max_work_group_size)
    return wg, blocks
end

kernelfor(k, ::Nothing, backend) = k(backend)
kernelfor(k, group, backend) = k(backend, group)

"""
    kernelfor(d::Dispatch, device)

Build the dispatch's KA kernel and refuse a kernel that captures variables: a
plan resolves arguments once, so a captured array would go stale.
"""
function kernelfor(d::Dispatch, dev::Device)
    obj = kernelfor(d.kernel, d.group, backend(dev))
    isempty(fieldnames(typeof(obj.f))) ||
        throw(ArgumentError("dispatch!: the kernel closes over $(fieldnames(typeof(obj.f))). " *
                            "A plan resolves its arguments once, so a captured device array " *
                            "would be the one it had when the plan was built — pass it as a " *
                            "dispatch argument instead, where a rename is followed."))
    return obj
end

"""
    callgroup(kernel, group) -> group or nothing

The workgroup size a launch must still pass at the call for a `kernel` built by
[`kernelfor`](@ref). `nothing` for a `StaticSize` kernel, since passing it again
keys a second iteration plan; `group` for a `DynamicSize` kernel, which would
otherwise silently get KA's default size (a `Val(256)` body run in groups of 64
writes one element in four).
"""
callgroup(::KernelAbstractions.Kernel{<:Any,<:KernelAbstractions.NDIteration.StaticSize},
          group) = nothing
callgroup(::KernelAbstractions.Kernel, group) = group

"""
    defaultgroup(ndrange, maxthreads; threads = 64) -> Dims

The workgroup for a `DynamicSize` kernel whose launch names none, on every
backend: `threads` work-items (at most `maxthreads`) filling the dimensions in
order. Small groups bound the idle lanes in a row's last group (first-axis
maximal groups made a 1080x1920 frame 1.6x slower), and filling later axes keeps
a `(3, n)` range from idling most lanes.
"""
defaultgroup(ndrange::Tuple, maxthreads::Integer; threads::Integer = 64) =
    KI.threads_to_workgroupsize(min(threads, maxthreads), ndrange)

# `bake` turned a `DeviceRange` into its ceiling or a `DeferredRange`, so no
# indirect command is needed (`PassPlan.indirect`).
devicesized(::Launch) = false

"""
    Call(f, args, device)

A compiled call: the function and its declared arguments. The library launches
its own work, so `argsize` and `indirectindex` are zero and `devicesized` false.
"""
struct Call{F,A<:Tuple,D}
    f::F
    # Declared, not resolved: an arena may move after baking and a call has no
    # patchable argument memory, so operands are resolved at each call.
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
    args = argmap(a -> resolve(dev, a), d.args)
    # Refused here for every backend, per `runscalls`.
    if iscall(d)
        runscalls(dev) || throw(ArgumentError(
            "dispatch!: `$(d.kernel)` was declared as a call (no ndrange), and " *
            "$(typeof(dev)) cannot run one: its `run!` submits a recording it " *
            "built as a command buffer, and a host call cannot be written into " *
            "one. Declare the operation as dispatches instead — on this backend " *
            "a GEMM is `coopmat_gemm!`, which records like any other dispatch."))
        return Call(d.kernel, d.args, dev)
    end
    # A `KI.Kernel` takes the same `ndrange`/`workgroupsize` keywords as a
    # `KA.Kernel`, so `Launch` serves both.
    buildskernel(d.kernel, be) || return Launch(kikernel(d.kernel, dev, args), args,
                                                bakedrange(c, d.ndrange), d.group, dev)
    let k = kernelfor(d.kernel, d.group, be)
        Launch(k, args, bakedrange(c, d.ndrange), callgroup(k, d.group), dev)
    end
end

bakedrange(::Compile, n) = n

# With `max`, dispatch the ceiling and let the kernel's required bounds check
# drop the surplus; same-queue ordering needs no host sync. A `DeferredRange`
# syncs the device before every dispatch (320 per frame took 97% of a Hikari
# frame on M5), so it is used only without `max`.
bakedrange(c::Compile, r::DeviceRange) =
    r.max === nothing ? DeferredRange(resolve(c.graph.dev, r.count)) : r.max

"""
The `Pipelines` phase for every backend: assign each draw and dispatch its
argument block (`argcursor`) and each device-sized dispatch its indirect slot
(`indcursor`), check render areas, and let the backend compile each through
`compiledraw` and `compile_dispatch`.
"""
function run!(::Pipelines, c::Compile)
    argcursor = 0
    indcursor = 0
    for p in ordered(c)
        checkrenderarea(p)
        cds = CompiledDraw[]
        for d in p.draws
            cd = compiledraw(c, p, d, argcursor)
            # Distinct draw pipelines, for `npipelines`; dispatches are not counted.
            push!(c.pipelines, cd.compiled.pipeline)
            push!(cds, cd)
            argcursor += argalign(argsize(cd))
        end
        # Dispatches and traces compile to different types; `identity.` narrows
        # a single-kind pass to a concrete vector so the frame loop specialises.
        compiled = Any[]
        for d in p.dispatches
            ind = d.ndrange isa DeviceRange ? (indcursor += 1) : 0
            # Kernels `@eval`ed while declaring the graph (e.g. the
            # `gemv_ncontig_kernel` family) are newer than the caller's world.
            cd = Base.invokelatest(compile_dispatch, c, d, argcursor, ind)
            push!(compiled, cd)
            argcursor += argalign(argsize(cd))
        end
        imgs, pre, barrier = passbarriers(c, p)
        push!(c.passes, PassPlan(p, cds, identity.(compiled), imgs, pre, barrier))
    end
    return c
end

"""
    checkrenderarea(pass)

Throw unless every attachment of a render pass has the same extent; otherwise
part of the larger one would silently go unrendered.
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

"""Bytes a compiled draw or dispatch packs into the plan's argument memory; zero
for a `Launch`, which binds its arguments directly."""
argsize(d::CompiledDraw) = d.argsize
argsize(d::CompiledDispatch) = d.argsize
argsize(d::CompiledTrace) = d.argsize
argsize(::Launch) = 0

"""The indirect-command slot a compiled dispatch owns; zero when host-sized and
for a `Launch`."""
indirectindex(d::CompiledDispatch) = d.indirect
indirectindex(d::CompiledTrace) = d.indirect
indirectindex(::Launch) = 0

# A `DeviceRange` dispatch reads an indirect command that a core prepare kernel
# writes from the device count; backends only launch it (`emitkernel!`) and
# place the barrier after it (`emitpreparebarrier!`).

"""
    emitprepares!(emitter, plan, passplan)

One fused prepare kernel writing the indirect command of every device-sized
dispatch and trace in the pass, then one barrier, so the dispatches are not
serialised by a prepare each. The pass's `repeat!` gate is folded in: a
discarded iteration writes zero groups, which makes `repeat!` portable to
backends that cannot discard recorded work. `emitpass!` emits it before the
predicate scope so it also runs for discarded iterations.
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

# One thread writes all the pass's indirect commands, unrolled over the tuples
# (one kernel per arity).
@inline multiprepare!(::Tuple{}, ::Tuple{}, ::Tuple{}, go) = nothing
@inline function multiprepare!(inds::Tuple, sizes::Tuple, wss::Tuple, go)
    n = UInt32(@inbounds sizes[1][1])
    ws = wss[1]
    groups = go ? (n + ws - UInt32(1)) ÷ ws : UInt32(0)
    writegroups!(inds[1], groups)   # groupCountX, or the ray count of a trace
    multiprepare!(Base.tail(inds), Base.tail(sizes), Base.tail(wss), go)
    return nothing
end

"""
    writegroups!(slot, groups)

Inside a prepare, write one dispatch's workgroup count into the slot from
[`indirectslot`](@ref). The default writes the three-word indirect dispatch
command; Metal's ICB slots override it to rewrite the recorded command.
"""
@inline function writegroups!(ind, groups::UInt32)
    @inbounds begin
        ind[1] = groups
        ind[2] = UInt32(1)
        ind[3] = UInt32(1)
    end
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

"""
    drawviewport(pass, draw) -> viewport or nothing

The draw's viewport, else the pass's, resolved at compile so no draw inherits
the previous draw's rectangle. `nothing` means no draw in the pass sets one.
Throws when only some draws set one and the pass has no default, since the sign
of a full-target height is the caller's convention.
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
    packedargs(args, vargs, fargs)

The one argument list packed into the pipeline's single push constant range: a
[`DrawBinding`](@ref) always, since only it sees `rebind!`; otherwise the
non-empty list (or either, when both stages share one).
"""
@inline packedargs(b::DrawBinding, _, _) = b
@inline packedargs(::Tuple, vargs, fargs) = isempty(fargs) ? vargs : fargs

# Formats come from the pass's own targets: a pipeline is compiled for the
# attachments it draws into.
function compiledraw(c::Compile, p::Pass, d, argoff::Int)
    dev = c.graph.dev
    color_formats = [attachment_format(t) for t in p.targets]
    depth_format  = p.depth === nothing ? nothing : attachment_format(p.depth)
    # A cell's current contents; `rebind!` cannot change their types.
    vargs = argmap(a -> resolve(dev, a), drawargs(d.args))
    fargs = argmap(a -> resolve(dev, a), d.frag_args)
    compiled = compile_draw(dev, d.shader, color_formats, depth_format, vargs, fargs;
                            bindings = d.bindings)
    indices = d.indices === nothing ? nothing : resolve(dev, d.indices)
    # One list, not both: stages may share the same list.
    stageargs = packedargs(d.args, vargs, fargs)
    return CompiledDraw(compiled, d.shader, stageargs, bakedcount(c, d.count),
                        drawviewport(p, d), d.bindings, indices, d.instances, argoff, 0)
end
compile_dispatch(c::Compile, d::Dispatch, ::Int, ::Int) = bake(c, d)

"""A draw's count with an indirect `Commands` buffer resolved to its storage."""
bakedcount(::Compile, n) = n
bakedcount(c::Compile, n::Commands) = Commands(resolve(c.graph.dev, n.resource))

"""An attachment's element type, which is the format a pipeline compiles for."""
attachment_format(t) = eltype(t)

# Where a host store lands in a walked run: an in-place write on the calling
# thread at the update pass. Whether in-flight work still reads the memory is
# `awaitwrites`' concern; `landstores!` decides when a store leaves its cell.
landhost!(r::GPURef{T}, ptr::Ptr{T}) where {T} = (update!(r, unsafe_load(ptr)); nothing)
landhost!(b::Buffer, data::AbstractVector, from::Integer) =
    (update!(b, from:(from + length(data) - 1), data); nothing)
# A transient's storage is an array over its arena slice: copy in place.
landhost!(t::TransientBuffer, data::AbstractVector, from::Integer) =
    (copyto!(view(storage(t), from:(from + length(data) - 1)), data); nothing)

"""
    run!(plan)

Run `plan` once: check it is live, reclaim dropped resources, acquire surface
images ([`beforeframe!`](@ref)), refit, check extents, re-record a stale
recording, then [`execute!`](@ref). Never records a plan for the first time and
writes no argument memory; per-run values are [`GPURef`](@ref) stores in the
run's own submission. A failure after the acquire hands the image back
(`abandonframe!`).
"""
function run!(pl::Plan)
    checklive(pl, pl.slabs, length(pl.graph.transients))
    reclaim!(pool(pl.graph.dev), pl.graph.dev)
    beforeframe!(pl.graph.dev, pl)
    try
        # After the acquire, which is the last place a resize is noticed.
        # Refitting earlier can pair a new-size colour target with an old-size
        # depth target, a lost device on NVIDIA.
        refit!(pl)
        checkextents(pl)
        measure!(pl)
        # Re-record on the owning thread what a move or refit invalidated.
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

The per-frame preamble before refit: collect the profiler's last frame, update
each surface's window (`beginframe!`) and acquire its image.
"""
function beforeframe!(dev, pl::Plan)
    # Before this run overwrites them, without waiting. Only stamped runs have
    # timestamps; skipping avoids a dynamic call per run for measured plans.
    pl.stamped && collect!(pl.profiler, dev)
    for s in pl.graph.surfaces
        beginframe!(s.win)
        acquire_next_image!(s.win)
    end
    return nothing
end

"""
    execute!(device, plan)

One run of the plan. An unrecorded plan is walked into a run (always on
interpreted backends, and for plans that cannot be recorded); a recording backend
refuses an unrecorded recordable plan. A recorded plan opens a front run only for
pending host stores or address patches, then submits the recording with it.
"""
function execute!(dev, pl::Plan)
    rec = pl.recording
    tok = if rec === nothing
        recordsplans(dev) && recordable(dev, pl) && throw(ArgumentError(
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
    pl.stamped && (pl.profiler.pending = true)
    return nothing
end

"""
    emitupdates!(emitter, plan)

Emit the update passes' pending host stores in front of a recording, in this
run's submission. Recordings never contain them.
"""
function emitupdates!(e, pl::Plan)
    for pp in pl.passes
        pp.pass.kind === :update || continue
        emitupdate!(e, pl, pp)
    end
    return nothing
end

"""
    emitpatches!(emitter, plan)

Emit pending address patches from `notify_move!` in front of a recording: eight
bytes per moved pointer, inline like a `GPURef` store.
"""
function emitpatches!(e, pl::Plan)
    patches = lock(pool(pl.graph.dev).lock) do
        p = pl.pending_patches
        # Checked under the lock, which `notify_move!` appends under.
        isempty(p) ? nothing : (pl.pending_patches = Tuple{Any,Int,UInt64}[]; p)
    end
    patches === nothing && return nothing
    for (r, off, val) in patches
        ref = Ref(val)
        GC.@preserve ref emitinline!(e, r, off, Base.unsafe_convert(Ptr{Cvoid}, ref), 8)
    end
    return nothing
end

"""`emitstore!` bound to one emitter, for `landstores!`."""
struct StoreEmitter{E}
    e::E
end
(s::StoreEmitter)(r::GPURef{T}, ptr::Ptr{T}) where {T} = emitstore!(s.e, r, ptr)
(s::StoreEmitter)(b::Union{Buffer,TransientBuffer}, data::AbstractVector, from::Integer) =
    emitstore!(s.e, b, data, from)

"""
    emitstore!(emitter, ref, ptr)
    emitstore!(emitter, buffer, data, from)

One host store as a command in this run, via the backend's `storebytes!`: a
`GPURef` value read straight from its pending cell at `ptr` (no boxed copy), or
a buffer range from the vector `update!` kept.
"""
emitstore!(e, r::GPURef{T}, ptr::Ptr{T}) where {T} =
    storebytes!(e, r.store, 0, Ptr{Cvoid}(ptr), sizeof(T))
emitstore!(e, b::Buffer{T}, data::AbstractVector{T}, from::Integer) where {T} =
    storefrom!(e, b.store, data, from)
# The transient itself, not `storage(t)`: the backend needs its block to name
# the destination buffer.
emitstore!(e, t::TransientBuffer{T}, data::AbstractVector{T}, from::Integer) where {T} =
    storefrom!(e, t, data, from)

"""Store `data` at element offset `from` (1-based) of `target` via `storebytes!`."""
function storefrom!(e, target, data::AbstractVector{T}, from::Integer) where {T}
    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src storebytes!(e, target, (Int(from) - 1) * sizeof(T),
                                 Ptr{Cvoid}(pointer(src)), length(src) * sizeof(T))
    return nothing
end

"""
    timings(plan) -> Vector{PassTiming}

Median cost of each pass over the last [`NSAMPLES`](@ref) frames. `gpu_ms` is
`NaN` without timestamps. Requires `Plan(g; profile = true)`.
"""
function timings(pl::Plan)
    # A budget-measuring profiler only holds the first runs' samples.
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

# The one walk over a plan's passes, through the primitives in
# `graph/backend.jl`: `record!` walks it once into a recording; backends without
# recordings walk it every `execute!` with the `Immediate` emitter.

"""
    emitplan!(emitter, plan)

Walk the plan into `emitter`: the head, then every pass in compile order.
"""
function emitplan!(e, pl::Plan)
    emithead!(e, pl)
    # Loop bodies are walked once per iteration here; recordings instead cut a
    # loop into pieces (`partitionranges`) and repeat them at submission.
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
    emitpasses!(emitter, plan, range)

Emit the passes in `range`, with profiler timestamps around each when stamped.
[`recordparts!`](@ref) uses it to emit one slice per recorded piece.
"""
function emitpasses!(e, pl::Plan, range)
    # Two loops so an unprofiled run pays no per-pass branch.
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
    emitpass!(emitter, plan, passplan)

One pass: barriers, the fused prepare, then its work inside its predicate scope.
An update pass goes whole to `emitupdate!`, since the emitter decides where its
stores land.
"""
function emitpass!(e, pl, pp::PassPlan)
    p = pp.pass
    p.kind === :update && return emitupdate!(e, pl, pp)
    emitbarriers!(e, pp)
    # Before the predicate scope, so a discarded iteration still writes zero
    # groups or rays. Vulkan conditional rendering does not cover trace-rays, so
    # a trace would otherwise run with an unwritten ray count.
    p.kind === :compute && emitprepares!(e, pl, pp)
    # Barriers stay outside the scope, so a discarded iteration still orders
    # its neighbours.
    withpredicate(e, p.predicate) do
        emitwork!(e, pl, pp)
    end
    return nothing
end

"""The pass's own commands: dispatches, a copy, or a render pass with its draws."""
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
            # Even if a draw throws: an open render pass breaks the command
            # buffer and the next failure would name the wrong pass.
            endrender!(e, h)
        end
    end
    return nothing
end

"""
    record!(plan) -> plan

Write the plan's commands once so every `run!` submits the same recording (does
not run it). Everything is resolved now for the plan's life, except
[`GPURef`](@ref) values, written by the run's update pass behind a fixed address.
Throws for plans with a surface or a `DrawBinding`. A backend without command
buffers returns the plan unchanged and `run!` walks it. Submission cuts come
from [`partitionranges`](@ref).
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
    ranges = partitionranges(pl)
    # Timestamps while measuring, afterwards only if timings were requested.
    pl.stamped = pl.profiler !== nothing && (pl.profile || measuring(pl))
    e = openrecording(dev, pl, first(ranges))
    if e === nothing
        # No recording: `run!` walks the plan.
        pl.stamped = pl.profile
        return pl
    end
    # Refilled by `closerecording!` from every piece; read only by `notify_move!`.
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

Record each of [`partitionranges`](@ref)' ranges as its own piece, the head into
the first only. `emitter` is the already-open first piece (`recordplan!` opens it
to learn whether the backend records). On a throw every piece is released.
"""
function recordparts!(dev, pl::Plan, emitter, ranges)
    parts = Any[]
    e = emitter
    heads = Set(first(r) for (r, _) in pl.loops)
    try
        for chunk in ranges
            e === nothing && (e = openrecording(dev, pl, chunk))
            first(chunk) == 1 && emithead!(e, pl)
            # A loop body's first piece starts each iteration with a full barrier.
            isempty(chunk) || first(chunk) in heads &&
                emitloophead!(e, pl, only(r for (r, _) in pl.loops if first(r) == first(chunk)))
            emitpasses!(e, pl, chunk)
            push!(parts, closerecording!(e, pl))
            e = nothing
        end
    catch
        e === nothing || abandonrecording!(dev, e)
        foreach(release!, parts)
        empty!(pl.patchtab)
        rethrow()
    end
    # Narrow to the backend's piece type so `submitrecording!` specialises.
    parts = identity.(parts)
    return RecordingParts(parts, submissionbatches(pl, ranges, parts))
end

"""
    submissionbatches(plan, ranges, parts) -> batches

The pieces in submission order (a loop body's repeated per iteration), grouped
into submissions while their measured cost fits the budget. An unmeasured piece
is submitted alone. At a finite budget each iteration
of a gated loop starts a new submission, because its timestamps describe only
the last, possibly discarded, iteration.
"""
function submissionbatches(pl::Plan, ranges, parts::Vector{R}) where {R}
    loopof(r) = isempty(r) ? nothing : pl.passes[first(r)].pass.loop
    order = Int[]
    gatedheads = Set{Int}()
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
            if isfinite(pl.budget) && any(i -> pl.passes[i].pass.predicate !== nothing,
                                         first(ranges[k]):last(ranges[j]))
                push!(gatedheads, k)
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
    t = Inf
    for k in order
        c = cost(k)
        if k in gatedheads || isempty(batches) || t + c > pl.budget
            push!(batches, R[])
            t = 0.0
        end
        push!(batches[end], parts[k])
        t += c
    end
    return batches
end

"""
    partitionranges(plan) -> Vector{UnitRange{Int}}

The pass ranges of the recorded pieces. A piece ends where the next pass would
exceed the plan's `budget` of measured GPU time (unmeasured passes count as
unbounded, so an unmeasured plan is cut per pass and timed, since a
watchdog-killed submission fails silently; no cut without timestamps or with
`budget = Inf`), and at [`repeat!`](@ref) body edges. The first range carries
the head barrier and query-pool reset, so a graph opening on a loop gets an empty
leading piece.
"""
function partitionranges(pl::Plan)
    n = length(pl.passes)
    out = UnitRange{Int}[]
    n == 0 && return push!(out, 1:0)
    loop(i) = pl.passes[i].pass.loop
    budget = pl.budget
    timed = measurable(pl)
    cost(i) = timed ? (isnan(pl.passcost[i]) ? Inf : pl.passcost[i]) : 0.0
    loop(1) === nothing || push!(out, 1:0)
    i = 1
    while i <= n
        l = loop(i)
        j = i
        t = cost(i)
        while j < n && loop(j + 1) === l && t + cost(j + 1) <= budget
            j += 1
            t += cost(j)
        end
        push!(out, i:j)
        i = j + 1
    end
    return out
end

"""Whether the plan can time its passes to cut by a finite budget."""
measurable(pl::Plan) =
    pl.profiler !== nothing && !isnan(pl.profiler.period_ns) && isfinite(pl.budget)

"""Whether the plan still has an untimed pass."""
measuring(pl::Plan) = measurable(pl) && any(isnan, pl.passcost)

"""
    measure!(plan) -> plan

Store the worst sampled time of each untimed pass in `passcost` (a budget is a
bound, and first runs are often slowest). Once every pass is timed, invalidate the recording if the partition changes or its timestamps are
no longer wanted, so `run!` re-records once. A `repeat!` body's time is its last
iteration's, which `submissionbatches` accounts for.
"""
function measure!(pl::Plan)
    (pl.stamped && measuring(pl)) || return pl
    prof = pl.profiler
    for i in eachindex(pl.passcost)
        isnan(pl.passcost[i]) || continue
        s = prof.gpu_ns[i]
        isempty(s) || (pl.passcost[i] = maximum(s) * 1e-9)
    end
    # Re-cutting on a partial measurement would re-record twice.
    measuring(pl) && return pl
    pl.recording === nothing && return pl
    (partitionranges(pl) != pl.partition || (pl.stamped && !pl.profile)) && invalidate!(pl)
    return pl
end

"""
The recorded pieces of one plan (see [`partitionranges`](@ref)), so no
submission exceeds the budget or the driver timeout. Barriers between pieces are
the graph's, and pieces are submitted in order. Backends implement
`submitrecording!` for one piece; core submits the sequence.
"""
struct RecordingParts{R}
    parts::Vector{R}
    # Submissions in order, a loop body's pieces repeated per iteration
    # (`submissionbatches`); each goes to the backend as one vector.
    batches::Vector{Vector{R}}
end

release!(rec::RecordingParts) = (foreach(release!, rec.parts); nothing)

"""
    submitrecording!(ctx, recording, emitter) -> token

Submit a recording; backends answer for their own piece type, this method
submits a partition's batches in order. The run's emitter, carrying host stores
and patches, goes with the first submission.
"""
function submitrecording!(ctx, rec::RecordingParts, e)
    tok = submitrecording!(ctx, rec.batches[1], e)
    for k in 2:length(rec.batches)
        tok = submitrecording!(ctx, rec.batches[k], nothing)
    end
    return tok
end

"""
    submitplan!(ctx, plan, emitter, R) -> token

Submit what `plan` recorded, `R` being the backend's piece type. Narrows the
`Any`-typed `Plan.recording` to `R`, `RecordingParts{R}` or `nothing` so the
calls are static; a dynamic call would box the returned `UInt64` token every run.
"""
function submitplan!(ctx, pl::Plan, e, ::Type{R}) where {R}
    rec = pl.recording
    rec isa R && return submitrecording!(ctx, rec, e)
    rec === nothing && return submitrecording!(ctx, nothing, e)
    return submitrecording!(ctx, rec::RecordingParts{R}, e)
end

"""
    submitrecording!(ctx, pieces::AbstractVector, emitter) -> token

One submission of several pieces, in order. This default submits them one by one;
backends that batch command buffers specialise it.
"""
function submitrecording!(ctx, pieces::AbstractVector, e)
    tok = submitrecording!(ctx, pieces[1], e)
    for k in 2:length(pieces)
        tok = submitrecording!(ctx, pieces[k], nothing)
    end
    return tok
end

"""
    recordable(plan) -> Bool

Whether the plan's commands can be recorded once. Not with a surface (the
swapchain image changes per frame) or a [`DrawBinding`](@ref) (rewritten between
runs); such plans are walked every run and the packer re-reads every cell.
"""
recordable(pl::Plan) = isempty(pl.graph.surfaces) && !hasrebindable(pl)

"""
    recordable(device, plan) -> Bool

Whether `device` records this plan: core's conditions plus the backend's own. A
declined plan is walked by `run!` instead of refused.
"""
recordable(::Device, pl::Plan) = recordable(pl)

"""Whether any draw takes its arguments from a `DrawBinding`."""
hasrebindable(pl::Plan) = any(d -> d.args isa DrawBinding, (d for p in pl.graph.passes for d in p.draws))

# The `Immediate` emitter: primitives do their work at once, through the verbs in
# `graphics/commands.jl` and the baked dispatch callables.

function withpredicate(f, ::Immediate, (pred, i)::Tuple{Any,Int})
    # The gate dispatch already ran on this thread; read its flag directly.
    storage(pred)[i + 1].go == 0 && return nothing
    return f()
end

# Lands the pending host stores at the update pass, before every reader.
emitupdate!(::Immediate, pl::Plan, ::PassPlan) =
    (landstores!(landhost!, pl.hostwritten); nothing)

emitdispatch!(::Immediate, d, ::AbstractString) = (d(); nothing)

# Passes run one after another on the host; nothing to order.
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
    # `target_view`, not `storage`: a transient attachment binds its view.
    targets = Any[target_view(t) for t in p.targets]
    depth = p.depth === nothing ? nothing : target_view(p.depth)
    return begin_render_pass!(pl.graph.dev, targets, p.loads, depth, p.depth_load)
end
function emitdraw!(::Immediate, handle, d)
    d.viewport === nothing || setviewport!(handle, d.viewport...)
    # Read through the binding: textures, counts, instances and indices may all
    # change per run without rebuilding the plan.
    bind = boundbindings(d.args, d.bindings)
    bind === nothing || use_bindings!(handle, d.compiled, bind)
    record_draw!(handle, d.compiled, drawargs(d.args), drawcount(boundcount(d.args, d.count));
                 instances = boundinstances(d.args, d.instances),
                 indices = boundindices(d.args, d.indices))
    return nothing
end

"""
    drawcount(count)

The count passed to `record_draw!`: an integer or an indirect `Commands` as is,
anything else (a buffer drawn whole) by its [`count`](@ref) at record time.
"""
drawcount(n::Integer) = n
drawcount(c::Commands) = c
drawcount(x) = count(x)
endrender!(::Immediate, handle) = (end_render_pass!(handle); nothing)

"""
    emitinline!(::Immediate, argmemory, offset, ptr, n)

Write a patch directly through `ArgMemory`'s host pointer. Only valid when the
backend has already ordered runs (Metal drains the device in `openrun` for a
recorded plan); pipelining backends put the patch in the submission instead.
"""
function emitinline!(::Immediate, am::ArgMemory, off::Int, p::Ptr{Cvoid}, n::Int)
    unsafe_copyto!(am.ptr + off, convert(Ptr{UInt8}, p), n)
    return nothing
end
