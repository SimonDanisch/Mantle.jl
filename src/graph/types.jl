# The graph's data structures.
#
# All of this was `src/vulkan/graph.jl`, and none of it is Vulkan's. Of the 21
# types here, 17 held no driver object at all and 4 held exactly one, which is
# now a type parameter:
#
#     Graph{D}       dev        — the device, whatever a backend calls one
#     PassPlan{D,B}  dispatches, images — what a compiled dispatch is, and\n#                    the barriers a backend emits
#     ArgMemory{S}   store      — the buffer arguments are written into
#     Profiler{Q}    pool       — the timestamp query pool
#
# Two types stayed behind because they ARE driver objects: `ImageBarrier` and
# the window wrapper. `TransientImage` came across too — five of its eleven
# fields hold driver objects, and they are five more type parameters.
#
# `Graph` and `Plan` were `abstract type`s in `runtime/api.jl` with one concrete
# subtype per backend. They are concrete here, and that is the point of the
# move: there is one graph, Mantle compiles it, and a backend records and
# submits what it is given. A backend that needs a graph type of its own is a
# backend reimplementing the scheduler.

"""
The swapchain image is externally indexed: acquire returns whatever the
presentation engine chooses, not `frame % n`. So it is its own type, and `run!`
brackets the acquire, the wait and the present rather than any user code.
"""
struct WindowSurface <: Resource
    win::Any
end

struct DrawCall
    shader::Any
    args::Tuple
    count::Any
    frag_args::Tuple
end

"""
One hardware ray-tracing launch, declared the way a dispatch is.

`Dispatch` is a kernel, its arguments and an ndrange; this is a ray-tracing
pipeline, its arguments and a ray count. What differs is the two ends — a shader
binding table where a dispatch has a kernel, a device-side ray count where it has
an ndrange — and NOT the middle, which is the whole reason this type exists.

Before it, tracing went through `custom!` — the escape hatch for work "the graph
declares but does not model", since deleted. That was the wrong home, and the
cost was specific: a `custom!` body packed its own arguments while it ran, out of
the batch queue's per-run scratch, and pushed that address as a push constant.
The host writes those bytes — no GPU command in the buffer does — so a recorded
plan, which never runs the body again, submits a command buffer aimed at memory
the queue has since handed to somebody else. Not stale values; whatever the next
caller packed there.

(An indirect DISPATCH is fine in the same situation, and the difference is worth
stating: its indirect command is written by a prepare kernel that is itself
inside the recorded buffer, so each run re-executes it and rewrites its own
region.)

Modelled, the arguments live at a fixed offset in the plan's `ArgMemory` exactly
as a dispatch's do, written once at `record!` — so a hardware ray-tracing plan
can be recorded at all.
"""
struct Trace
    pipeline::Any
    accel::Any
    args::Tuple
    # A count, in the same two shapes `Dispatch` takes it: a plain number, or a
    # `DeviceRange` over a count only the GPU knows.
    ndrange::Any
end

# `draw!` does not take fragment arguments yet — Lava's `draw!` has taken a
# `frag_args` tuple for a while and this is the field it will arrive in. Empty
# is what the pipeline phase compiles against today.

mutable struct Pass
    name::String
    kind::Symbol
    targets::Vector{Any}                # render: the colour attachments. copy: the source.
    loads::Vector{LoadOp}        # render only, one per colour attachment
    depth::Any                          # render only, and only if one was given
    depth_load::Union{Nothing,LoadOp}
    dst::Any                            # copy only
    draws::Vector{DrawCall}
    usages::Vector{Pair{Int,Type}}
    dispatches::Vector{Any}
    # `nothing`, or the `(buffer, index)` whose 32-bit value at record time
    # decides whether this pass's WORK runs — see [`repeat!`](@ref). Its barriers
    # run either way, which is what makes a skipped iteration harmless to the
    # ones around it rather than a hole in the ordering.
    #
    # On the pass rather than on a scope enclosing several, because the graph has
    # no nesting: passes are a flat list that the scheduler reorders and the
    # placer allocates against, and a construct that owned a *range* of them
    # would be one more thing every rewrite has to keep consistent. A predicate
    # is a property of a pass, so it travels with it.
    predicate::Any
end

abstract type TransientResource <: Resource end

mutable struct TransientBuffer{T} <: TransientResource
    n::Int
    first::Int
    last::Int
    # Where in the pool this landed. The BLOCK is kept, not just the fused
    # address, because a Vulkan buffer barrier scopes to (VkBuffer, offset, size)
    # and an address alone cannot name the buffer.
    block::Any            # ::BufferBlock once placed
    offset::Int
end

"""
A transient render target.

The backend's image object exists from the moment the handle does, because its
memory requirements are what the placer needs and only a real image can be
asked for them. It costs no memory until bound, and the graph is built once, so
this is not a per-frame cost.

Five parameters because five of the fields ARE driver objects and none of them
is the graph's business: what a format, a usage mask, an image, and a memory
requirement are differs per backend, and only the ELEMENT TYPE `T` is portable.
They are parameters rather than `Any` for the reason `BatchQueue`'s ten are —
`target_view` and `target_image` are read while a pass is recorded, and an
untyped field there is a dynamic lookup per attachment per frame. A backend
gives itself a concrete alias (`VulkanTransientImage`) and its own code reads
the same.

`req` is whatever the backend needs to place and remake this image; the graph
reads `.size` and `.alignment` from it and nothing else.
"""
mutable struct TransientImage{T,F,U,I,R} <: TransientResource
    width::Int
    height::Int
    format::F
    usage::U
    image::I
    req::R
    first::Int
    last::Int
    view::Any
    memory::Any
    # What this target takes its size from, or nothing for a fixed size. A
    # window changes size under a plan, and a depth buffer that does not change
    # with it stops covering the render area.
    source::Any
end

"""
Which allocation a transient belongs in.

Buffers and images are placed separately rather than in one arena. Vulkan permits
mixing them, but only with `bufferImageGranularity` padding between a linear and
an optimal-tiled resource, and getting that wrong is aliasing corruption that no
validation layer reports. Two arenas cost one extra allocation and make the rule
unnecessary.
"""
struct Buffers end

struct Images end

"""
Bytes a RECORDING reads while it runs: host-writable, device-addressable, and
never moved while a recording that names them exists.

A recorded dispatch holds its argument block's address as a push constant and its
workgroup counts as an indirect command the command processor reads, so both have
to be memory the host writes and the device addresses. Neither may be handed to
anyone else for as long as the recording is replayable — which is a `Region`, and
is what the pool has been handing out all along.

It is an arena KIND rather than a bump allocator per queue because four of those
were written for want of one: an argument slab ring on the queue, an indirect
slab ring beside it, and a third copy of the first inside every capture. A kind
is what `acquire!`/`release!` need in order to answer the question, and the
lifetime then belongs to whoever owns the recording rather than to a cursor that
has to guess when the device is finished.
"""
struct Unified end

"""
Bytes the HOST reads after the device wrote them: a download's staging.

Its own kind, and not `Unified`, because of what the host does with the bytes.
`Unified` is BAR memory — device-local and host-visible — which the host WRITES
well and READS at write-combined speed: 29 MB/s on an RTX 4000 Ada and about
the same on a RADV RX 7900 XTX, measured 2026-09-05, against a few GB/s from
cached memory. A download that staged through it took 330 ms for one
1368x1026 RGBA{Float32} frame, a third of a 32-sample render. A backend
answers `rawalloc` for this kind with host-cached memory where the device has
it, and plain host-visible memory where it does not.
"""
struct Readback end

# `Recycler` is gone with renaming. It held retired buffers by byte size and
# handed them back once a completion token said the device was past the run that
# read them — the free list an `Update` took its fresh store from. An update
# writes its target IN PLACE now, so there is no outgoing store to hold and
# nothing to recycle.

# `ArgWrite` is gone, and with it the whole idea of a per-run host write into
# argument memory. It held "this `Ref`'s bytes go at offset N of this entry's
# argument region", one per (entry, `Ref`) pair — 602 of them on Hikari's fused
# sample, to move the 268 bytes that actually changed, because every dispatch
# carries its own COPY of every argument it is given.
#
# A [`GPURef`](@ref) removes the copies rather than making them cheaper to
# rewrite: the argument each dispatch is packed with is the ref's device
# ADDRESS, written once at `record!` and never again, and the value behind it is
# one buffer that one store (`ref[] = x`) writes. 602 becomes 1, and it becomes a command
# in the stream rather than a host store into memory a submission may still be
# reading — which is what let the argument ring go.

mutable struct Graph{D}
    dev::D
    passes::Vector{Pass}
    surfaces::Vector{WindowSurface}
    transients::Vector{TransientResource}
    transient_by_id::Dict{Int,TransientResource}
    ids::IdTable          # both directions; see `IdTable`
    # Interning for `use(...; range = ...)`. `ids` is an IdDict, so two `use`
    # calls naming the same slice would otherwise be two objects and two ids —
    # and a resource that is not the same resource in two passes has no hazards
    # between them, which is the one answer that must not be reachable by
    # accident. Keyed by value, so the same slice is the same sub-resource.
    views::Dict{Tuple{Int,UnitRange{Int}},Any}
end

struct PassHandle
    graph::Graph
    pass::Pass
end

"""
The binding a pass gets from an attribute. A `Buffer` and a `GPURef` of the same
element type both erase to `Attr{T}`, differing only in `stride`, which is a
field rather than a type parameter.

That is what makes a scalar and a per-element attribute one pipeline: the binding
tuple is the pipeline key, so if the two spellings produced different types they
would produce different shaders.
"""
struct Attr{T,R} <: Resource
    resource::R
    stride::Int32
end

"""A draw count that lives on the device, wrapped so recording dispatches on it."""
struct Commands{R}
    resource::R
end

"""
A slice of a buffer, as its own resource.

Two passes writing disjoint halves of one buffer do not race, and tracking the
buffer whole says they do — a barrier between them orders memory neither touches.
Handing the slice its own id is what lets the existing per-resource walk answer
that without knowing anything about ranges: disjoint slices are disjoint
resources, and the hazard set falls out unchanged.

`range` is in elements, and the barrier is scoped to exactly those bytes. The
kernel still receives the whole buffer — a range declares what a pass *touches*,
not what it can address.
"""
struct BufferRange
    parent::Any
    range::UnitRange{Int}
end

struct CompiledDraw{P,S,A<:Tuple,C}
    # Concrete throughout: `Any` here made every field access in the per-draw
    # record path a dynamic lookup, which is where the frame's allocations were.
    # `P` and `S` are what the backend compiled — a `VulkanCompiledGraphicsPipeline`
    # and a `LavaGfxShader` on Vulkan. Parameters rather than `Any` precisely
    # because of the sentence above: the point is that they stay concrete.
    compiled::P
    shader::S
    args::A
    count::C
    # Where this draw's arguments live inside a slot of the plan's argument
    # memory. Fixed at compile time, because the set of draws and the size of
    # each one's arguments are.
    argoff::Int
    argsize::Int
end

"""
One dispatch, resolved the way a draw is: the shader is compiled when the plan
is, and a frame only writes arguments and records.

Going through `KernelAbstractions` per frame instead looked equivalent and was
not. Its launch plan is keyed on `Base.get_world_counter()`, so defining a method
anywhere — every eval in a session — misses the cache, and the miss is an
unconditional recompile that shells out to `spirv-opt`. That is ~20 ms, and
because waiting on a subprocess is a task switch it happens *between* the
swapchain acquire and the present. The draws never had this: they hold their
pipeline. Now the dispatches do too, and the only thing that compiles a kernel is
building a plan.

`kernel` is the kernel function after `Adapt`, which for a `@kernel` is a
singleton — checked when the plan is built, because a closure over device arrays
would be resolved once here and then go stale the first time an argument is
renamed.
"""
struct CompiledDispatch{L,K,A<:Tuple,I,N,R,O}
    # Concrete throughout, for the reason `CompiledDraw` is: an `Any` field turns
    # every access in the per-dispatch record path into a dynamic lookup. `L` is
    # the backend's launch plan.
    launch::L
    iter::I                             # IterPlan for `nd0`
    nd0::N                              # the ndrange the iteration plan was built for
    obj::O                              # the KA kernel, for an ndrange that moves
    kernel::K
    args::A
    ndrange::R
    tlas::Bool                          # whether the pipeline was built for ray query
    argoff::Int
    argsize::Int
    # Which of the plan's indirect commands this dispatch reads its workgroup
    # counts from, or 0 when the ndrange is the host's. Assigned at compile beside
    # `argoff` and for the same reason: the plan knows how many device-sized
    # dispatches it has, so each gets a fixed place in the plan's own memory
    # rather than a slot from an allocator that has to be told when to rewind.
    indirect::Int
end

"""
A [`Trace`](@ref) after the backend has resolved it — the compiled counterpart of
[`CompiledDispatch`](@ref), and deliberately the same shape.

`compiled` is whatever the backend needs to record the launch: on Vulkan a
`VulkanTracePipeline`, which is the `VkPipeline`, its shader binding table, and
the argument layout the raygen shader was compiled for. Core reads exactly one
thing out of this type, `argsize`, which is what lets `ArgMemory` reserve a slot
for the trace's arguments the same way it does for a dispatch's.

`accel` and `args` are held as GIVEN, not as resolved: `rawargs` resolves them
to their storage at `record!`, once. A value that changes between runs — a
sample index — is a [`GPURef`](@ref), and what is packed is its address.
"""
struct CompiledTrace{P,C,A<:Tuple,R}
    compiled::P
    accel::C
    args::A
    ndrange::R
    argoff::Int
    argsize::Int
    indirect::Int            # as `CompiledDispatch.indirect`; 0 for a host ndrange
end

"""
Argument memory owned by the plan: every draw's and dispatch's arguments, laid
out once, and the indirect commands beside them.

The plan knows its draws and their argument sizes at compile time, so it lays
them out once and never allocates per frame. **Written once, at `record!`, and
never again** — which is what makes one copy enough. A value that has to change
between runs is a [`GPURef`](@ref): the bytes here are its device address, fixed
for the ref's life, and the value behind that address is written by an `Update`,
as a command in the stream rather than a host store into memory a submission may
still be reading.

`token` is what covers the last run of this plan, or `nothing` before the first.
Opaque here and handed straight back to `passed`/`waitfor` — it was a raw Vulkan
timeline value, which made a plan's pipelining a Vulkan concept.

`store` is the [`Unified`](@ref) region the plan owns. It was a device array from
the backend's own allocator, which put the plan's arguments outside the one pool
that is supposed to see every workload.

**The indirect commands are in here too**, past the arguments, one 256-byte block
per device-sized dispatch. They belong to the plan for exactly the reason the
arguments do — the plan knows at compile how many it has, and a recording bakes
the address of each into a `vkCmdDispatchIndirect` — and putting them anywhere
else is what the queue's third bump allocator was.

`indirect[k]` is dispatch `k`'s view, built once here because building one per
record is an allocation on the recording path and the offsets never move.

This was a RING, `ARG_SLOTS` deep, with a `slot_token` per slot and a
`nextslot!` that waited when the host got three runs ahead. It existed for one
reason: a run wrote argument bytes on the HOST while an earlier run's submission
could still be reading them. Nothing writes these bytes after `record!` any
more, so there is nothing to race and nothing to rotate — and a plan's argument
memory is a third of the size it was.
"""
mutable struct ArgMemory{S}
    store::S
    address::UInt64
    ptr::Ptr{UInt8}
    # A `RefValue`, not a bare UInt64: `ArgMemory` is parametric and a plan's
    # `args` field names the UnionAll, so a bare value field reads and writes
    # through the DYNAMIC get-/setproperty path, which boxes the UInt64 — 8
    # bytes each way of a `run!` that allocates nothing. The cell is a
    # reference (no box) and the deref is on the concrete `RefValue`.
    # 0 = never run (the first real timeline signal is 1).
    token::Base.RefValue{UInt64}
    indirect::Vector{Any}       # [k] — one indirect command each
end

# Accessed through these, never as `am.token` on a value typed
# `Union{Nothing,ArgMemory}`: the field is a UnionAll, and get-/setproperty on
# it goes through the dynamic path, which boxes the UInt64 — 8 bytes on both
# sides of every `run!`. A method specializes on the concrete `ArgMemory{S}`
# and the field access inside it is static.
"""The token covering `am`'s last run; 0 if it never ran or has no memory."""
argtoken(am::ArgMemory) = am.token[]
argtoken(::Nothing) = UInt64(0)

"""Record the token covering `am`'s next submission."""
setargtoken!(am::ArgMemory, v::UInt64) = (am.token[] = v; nothing)

"""
One pass, its compiled draws, and the barriers that have to run before it.

Both are derived. The swapchain image arrives in whatever state acquire left it
and the pass needs it as a colour attachment, so that transition falls out of the
usage sequence rather than being written by hand.
"""
struct PassPlan{D,B}
    pass::Pass
    draws::Vector{CompiledDraw}
    # `D` is what a compiled dispatch IS on this backend: a `CompiledDispatch`
    # where the backend records commands, a callable where it drives
    # KernelAbstractions. Same slot, and parameterised rather than `Any` so the
    # per-frame loop over it still specialises.
    dispatches::Vector{D}
    images::Vector{B}        # layout changes, one barrier each
    pre::Vector{Transition}      # what this pass needs before it runs
    barrier::Any                        # one memory barrier covering the rest
    # Whether any of this pass's dispatches sizes itself on the device. Decided
    # at compile because it decides how the pass is recorded, and asking every
    # dispatch per frame is a dynamic call per element.
    indirect::Bool
end

"""
Two timestamps per pass and a ring of samples, or nothing at all.

A profiler is a plan-level thing rather than a device-level one because a pass is:
Lava's own timing is per dispatch and keyed by kernel name, which cannot see a
render pass, cannot see an update, and cannot tell two dispatches of one kernel
apart. Slots are `2i-1, 2i` for pass `i` in scheduled order.

`pending` says a frame's timestamps have been written and not yet read. They are
read without `WAIT_BIT`: a frame still in flight is skipped rather than waited
for, so turning profiling on does not change the frame time it reports.
"""
mutable struct Profiler{Q}
    pool::Q
    period_ns::Float64
    nslots::Int
    names::Vector{String}
    host_ns::Vector{Vector{Float64}}    # one ring per pass
    gpu_ns::Vector{Vector{Float64}}
    pending::Bool
end

"""The emitter of a backend without recordings: every primitive of the pass walk
does its work on the spot. See `graph/kalaunch.jl`."""
struct Immediate end

mutable struct Plan{D,H<:Tuple}
    graph::Graph{D}
    transitions::Vector{Transition}
    passes::Vector{PassPlan}
    pipelines::Set{Any}
    slabs::Vector{Any}                  # the SHARED region of each arena, not owned
    arenas::Vector{Any}                 # …and which arena each one is, for remap!/free!
    offsets::Vector{Int}                # per transient, relative to its region
    peak::Int
    naive::Int
    profiler::Union{Nothing,Profiler}
    alias::Bool                         # kept, so a recompile means what the first one did
    coalesce::Bool
    policy::Policy
    # `nothing` on a backend that has none: argument memory exists because a
    # RECORDED launch reads its arguments from a buffer the host wrote. A
    # backend driving KernelAbstractions passes them directly — see
    # `makeargmemory`.
    args::Union{Nothing,ArgMemory}      # laid out at compile, written at record!
    # The plan's commands, written once, or `nothing` before `record!`.
    #
    # ONE. It was a vector, one per argument slot, because a run rewrote argument
    # bytes on the host and had to write them somewhere the device was not
    # reading — so a recording belonged to the slot whose base offset was folded
    # into every address it held. Nothing rewrites those bytes now (see
    # [`ArgMemory`](@ref) and [`GPURef`](@ref)), so there is one set of addresses
    # and one command buffer holding them.
    recording::Any
    # Where every device address the recording's arguments hold was written:
    # recorded address → the (region, offset inside it) of its eight bytes.
    # The region is usually the plan's `ArgMemory` store, but a prepare kernel
    # emitted into the recording packs into a recording-OWNED scratch region
    # (`get_arg_buffer(recording, …)`), so the target is a region, not a blob
    # offset. Baked at `record!` — the graph knows the args and the layout, and
    # this is derived from them ONCE, during the one pack — and consulted only
    # when a resource MOVES (`resize!`, an arena growing): the move appends to
    # `pending_patches`, and the next `run!` writes the new addresses as
    # commands in the run's own submission. Neither is ever walked per run.
    # See `notify_move!`.
    patchtab::Dict{UInt64,Vector{Tuple{Region,Int}}}
    pending_patches::Vector{Tuple{Region,Int,UInt64}}
    # Every `Buffer` and `GPURef` a pass declares, as a tuple typed by its
    # elements: what a host store (`ref[] = x`, `buf[r] = data`) can be waiting
    # on. `run!` walks it before submitting — a dirty one lands as a command in
    # front of the recording, a clean one costs a flag read. Built once at
    # `Plan` by `hostwritten!`, which also registers each as a `CopyDst` of the
    # update pass so the barriers derive; a `refit!` recompiles against the same
    # registrations and the same tuple.
    hostwritten::H
    # A recording was thrown away by something that was not this plan's doing —
    # an images arena growing under it, a refit — and `run!` writes it again
    # before it submits. Never set for a plan that was not recorded: `run!`
    # refuses those rather than recording on the way past.
    stale::Bool
end

"""
The state a compilation carries between phases. Every field is written by exactly
one phase and read by later ones, which is what lets a phase be run alone.
"""
mutable struct Compile{D} <: Compilation
    graph::Graph{D}
    alias::Bool
    coalesce::Bool
    policy::Policy
    # What the backend-independent phases compute. Held rather than spread over
    # fields here, so those phases can live in core and read one thing.
    analysis::Analysis
    transitions::Vector{Transition}            # Barriers
    prepass::IdDict{Pass,Vector{Transition}}
    passes::Vector{PassPlan}                          # Pipelines
    pipelines::Set{Any}
end

"""
The buffer arena's block: one raw `vkAllocateMemory` with a `VkBuffer` bound into
it, and the buffer's device address.

Deliberately NOT a `LavaArray`. That would put Mantle's suballocation on top of
Lava's pool — two allocators with the lower one invisible — and it would drag in
a `DataRef` whose finalizer frees memory Mantle owns. Nothing here finalizes and
nothing refcounts: the Block owns the memory, the Pool owns the Block, `trim!`
frees.
"""
struct BufferBlock
    buffer::Any
    memory::Any
    address::UInt64
    bytes::Int
    # A `DataRef` whose releaser DOES NOTHING. This is Lava's shape of the
    # `unsafe_wrap(..., own = false)` hatch every GPU array package provides for
    # foreign memory: a transient's `storage` is a `LavaArray` view over it, so
    # host-side operations (copies, library calls) work — while the free stays
    # Mantle's, because the Block owns the memory and `trim!` frees.
    ref::Any
end
