# Arrays that change, pending changes, acceleration structures, particles:
# pseudocode.
#
# Not runnable. Four parts:
#   A. the user-facing API (application code, Makie recipes, Hikari scenes);
#   B. core internals;
#   C. the verbs a device type implements for it;
#   D. the consumers built on A: Raycore's software TLAS, Hikari, RayMakie.
# This file owns array storage, resize!, stores, pending changes and their
# application, dependents, and acceleration structures. internals.jl owns the
# rest (graphs, compile, record, run, ordering, memory, holds) and points here.
#
# Rulings this implements (Simon; decisions.md B2 and D, 2026-09-30; B3.2
# and B3.7, 2026-10-02):
# - Arrays are fixed until their first resize!; pointer and size reach
#   recorded work through the array's cell. Only what a plan recorded by value
#   recompiles it.
# - Changes (resize!, setindex!, copyto!, copy!, TLAS changes) update the host
#   view at the call and stay pending on the resource until the next
#   submission that touches it, which applies them in front of its own work.
# - "Kernel launch" means any launch of GPU work: traces count.
# - TLAS (B3.2): builds use the exact count, storage has a doubling capacity
#   that never halves, a plan's build or refit is an AccelCommand, Metal reads
#   the count from a cell, Vulkan's indirect build is not used.
# - GPURef of an array, attribute(a, i), the scene rectangle (B3.7).
# Eviction and readers pending structure updates are proposals (plan, open
# decision 16).
#
# Facts this rests on (checked 2026-09-30/10-02):
# - Inactive instance = BLAS reference 0; invisible, still counted in instance
#   ids; active <-> inactive needs a full build, not an update; an update may
#   not change the instance count (vkdocs chapters/accelstructures.adoc:255-300,
#   169-175). An inactive triangle has a NaN first vertex coordinate (same
#   place). An update must keep the geometry count, flags, type, vertex format,
#   maxVertex, index type and transform presence of the last build; which
#   BLAS an instance references is not among them
#   (chapters/commonvalidity/build_acceleration_structure_common.adoc:121-200).
# - Acceleration structure storage must not be a sparse-residency buffer
#   (resources.adoc:8748-8750, VUID-VkAccelerationStructureCreateInfoKHR-buffer-03615).
# - A build records the source and destination AS handles
#   (accelstructures.adoc:764-769).
# - accelerationStructureIndirectBuild: false on RADV 26.2.3 and NVIDIA 595.99,
#   true on AMD's Windows driver 26.7.1 (vulkaninfo; the Windows value from a
#   remote machine). Metal reads the instance count from a buffer:
#   MTL4IndirectInstanceAccelerationStructureDescriptor.instanceCountBuffer /
#   maxInstanceCount (MTL4AccelerationStructure.h:591-627, macOS SDK; Metal.jl
#   lib/mtl/libmtl.jl:3591-3597). MTL4 queues take residency sets
#   (MTL4CommandQueue.h:317-349).
# - sparseResidencyBuffer: true on RADV, NVIDIA, AMD Windows, lavapipe; sparse
#   address space (a device-wide total, resources.adoc:103-112) 128 TiB RADV,
#   1 TiB NVIDIA, about 128 TiB AMD Windows, 2 GiB lavapipe. Sparse pages are
#   bound at VkMemoryRequirements::alignment granularity (sparsemem.adoc:203-206).
#   Metal placement-sparse buffers validated on the M5 (2026-09-25).
# - Shaders can take a TLAS as a 64-bit address (OpConvertUToAccelerationStructureKHR,
#   SPV_KHR_ray_query; mentioned in vkdocs proposals/VK_EXT_mutable_descriptor_type.adoc:121).
#   Lava binds a descriptor today (Lava compiler/spirv/rayquery.jl:47,66-67);
#   which one Mantle uses is open decision 4 in the plan.
# - Descriptors may be written while a command buffer using the set is
#   pending only if that buffer does not use them (descriptorsets.adoc:768-807).
# - A kernel that reads a cell directly instead of its argument slot pays about
#   195-220 ns per small dependent dispatch on RADV and NVIDIA (v2_*.txt:
#   "headers read-only, indirect" minus "flat read-only, indirect").
# Measured 2026-10-01 (experiments/tlas_sparse/: m1_lapwin.txt, m23_lapwin.txt,
# bench_legionboi.txt; AMD Radeon 8060S, Windows driver 26.7.1, and RTX 3070
# Laptop, NVIDIA 595.91.07; GPU timestamps, validation clean on small runs).
# N active instances, capacity C, inactive past N, prefer-fast-trace:
# - Build time grows with C. N=100000: exact 1.08 ms AMD / 0.74 ms NVIDIA;
#   C=2N 1.41 / 0.84, C=4N 1.94 / 1.20, C=16N 5.41 / 3.42 ms. N=1, C=131072:
#   341 vs 12 µs AMD, 256 vs 33 µs NVIDIA.
# - Refit: AMD independent of C (N=100000: 256-264 µs at every C); NVIDIA
#   grows with C (166 µs exact, 243 at 2N, 393 at 4N, 1295 at 16N).
# - Compacted size padded/exact: AMD 1.02 (2N), 1.06 (4N), 1.29 (16N);
#   NVIDIA 1.8, 3.5, 14: NVIDIA keeps the inactive instances in the structure.
# - Count read from memory (AMD Windows, indirect build, maximum C): cheaper
#   than padding, slower than exact; N=100000: 1.08 ms exact, 1.17 ms at 2N,
#   1.25 at 4N, 1.94 at 16N.
# - Sparse binding: page 64 KiB on both. vkQueueBindSparse returns after
#   15-18 µs AMD, 90-96 µs NVIDIA; bound (fence) after 144 / 230 µs for one
#   page, 278 / 263 µs for 256, 1.86 / 0.83 ms for 4096 (256 MiB). Device
#   address unchanged, contents correct, never-bound pages read zeros.
# - One vkQueueSubmit costs 23.5-24 µs of host time on both: 1000 submissions
#   of one 16-byte update take 24 ms; one submission of all 1000 updates
#   completes after 152 µs AMD, 77 µs NVIDIA.
# - Traversal (m4_*.txt; 1M ray queries, the same rays through an exact and a
#   padded TLAS): padding costs nothing measurable, up to C=16N and N=100,
#   C=131072 (padded/exact within -1.3%..+1.7% NVIDIA, -0.5%..+0.5% AMD).
#   Every ray hits the same instance in both: no inactive instance is hit.

# ══ A. User-facing API ════════════════════════════════════════════════════════

# ── Arrays ────────────────────────────────────────────────────────────────────

pos = MantleArray(dev, rand(Point3f, 1000))     # fixed: launches over it are direct
# Changes update the host view at the call and are applied in front of the
# next submission that touches the array (a run, an eager call, a host read).
resize!(pos, 5000)                              # the first resize! makes it resizable (B.5)
pos[1:3] = newpoints                            # a store; lengths must match, as in Base
copyto!(pos, newdata)                           # a store
copy!(pos, newdata)                             # Base.copy!: resized to length(newdata) (old contents not kept), then stored
trim!(pos)                                      # (name open) unmaps a reserve's pages beyond the size; nothing else does
length(pos)                                     # the host view
devicesize(alive)                               # (name open) the length a kernel wrote: reads the cell back

# In a graph launches name the array, not a count.
g = Graph(dev)
dispatch!(g, advect!, (pos, vel, dt), launchrange(pos))
render!(g, target) do p
    draw!(p, pointshader, (pos, colors, cam), launchrange(pos))
end
run!(g)
copy!(pos, rand(Point3f, 3000)); run!(g)        # the run applies the change and copies the new length

# A length a kernel decides needs a capacity: no host call says how much to
# reserve, and graphs size temporaries from it.
alive = MantleArray(dev, Point3f, 0; capacity = 1_000_000)
dispatch!(g, spawnkill!, (alive, lengthof(alive), pos), launchrange(pos))

# ── Acceleration structures ───────────────────────────────────────────────────
#
# One core type for hardware and software ray tracing. TLAS(dev) picks the
# hardware path where the device has it; `hardware = false` forces software.

blas = BLAS(dev, vertices, faces)               # vertices::MantleArray{Point3f,1}, faces::MantleArray{TriangleFace{UInt32},1}
tlas = TLAS(dev)                                # storage sized for 64 instances, doubling; builds use the exact count (B.8);
                                                #   pends a Build of zero instances, so it is never traced unbuilt

h  = push!(tlas, blas, transform)               # one instance (transform::Mat4f)
r1 = push!(tlas, blas, transforms)              # a range, one instance per element (transforms::MantleArray{Mat4f,1})
r2 = push!(tlas, marker, positions;             # a range computed on the device from positions (meshscatter);
           scale = sizes, rotation = rotations) #   positions may be an array GPURef; sizes, rotations: arrays or one value
tlas[h] = newtransform                          # a store into the range's source: a refit
visible!(tlas, r2, false)                       # (name open) mask 0 in the range's records: a refit
delete!(tlas, r1)                               # pending: its slot is freed, a build follows
build!(tlas)                                    # pending: a full build, applied by the next use

# Per-run work inside a graph: an operation like any other.
build!(g, tlas)                                 # Build() every run: needed when a kernel decides counts
refit!(g, tlas)                                 # Update() every run; changes of the active set pend a build
trace!(g, pipeline, tlas, (film, cam), (w, h))  # ray tracing pipeline
dispatch!(g, shade!, (tlas, film), (w, h))      # ray queries in a compute kernel

# Drawn attributes: one argument type for one value and one per element.
pos   = GPURef(dev, MantleArray(dev, points))
color = GPURef(dev, MantleArray(dev, [RGBAf(1, 0, 0, 1)]))         # one value for every point
copy!(color[], fill(RGBAf(0, 0, 1, 1), length(points)))              # now one per point: a resize and a store
color[] = MantleArray(dev, other_colors)                              # another device array: a rebind, no copy
# In a shader: attribute(color, i) == color[min(i, length(color))] (1-based)

# Per-instance data in shaders: instanceCustomIndex is the range's slot (stable
# for the range's life); the instance's index within its range is
# InstanceId - rangeoffset(tlas, slot), read from an array the TLAS passes.

# ── Textures ──────────────────────────────────────────────────────────────────

atlas = Texture2D(dev, RGBA{N0f8}, 1024, 1024)  # has an index in the device's texture table
resize!(atlas, 2048, 2048)                      # a new image under a new index (B.9)

# ══ B. Core internals ═════════════════════════════════════════════════════════

# ── B.1 Storage ───────────────────────────────────────────────────────────────

struct Region                           # a pool region: fixed arrays, small resizable ones, AS storage
    block::Block; offset::Int; bytes::Int
end
mutable struct Reserve                  # address space with pages mapped as the array grows
    const native::Any                   # sparse VkBuffer (SPARSE_BINDING|SPARSE_RESIDENCY),
                                        # placement-sparse MTLBuffer, cuMemAddressReserve/hipMemAddressReserve
    const bytes::Int
    mapped::Int                         # bytes with pages behind them (host view, B.4)
    pages::Vector{PageRun}
end
struct Committed                        # a pool region with room to grow (doubling)
    region::Region
end
struct Released end                     # after the last hold (internals.jl section 10)

# Where a resizable array goes, by dispatch on what the device has and how big
# the array is. Decided by the submitter that applies a Resize (B.4 step 1),
# from the storage at that point of the sequence, never at the call. The
# first resize! keeps the array where it is if the new size
# fits (the region becomes Committed in place: only a cell write). Beyond its
# storage, a large array on a device with sparse residency moves into a
# reserve, where later growth maps pages and the address stays; any other
# array moves into a pool region of twice the need, and each later doubling
# is a move (B.6). A sparse page is VkMemoryRequirements::alignment of the
# sparse buffer (64 KiB on AMD Windows and NVIDIA, measured above). One bind
# costs 0.15-0.25 ms before the submission that uses the pages (measured
# above), so a reserve maps at least twice what it had: an array growing by
# one element per run binds O(log n) times.
growth(dev, r::Region, need)    = need <= sizeof(r) ? Adopt() : outgrow(caps(dev), need, caps(dev).sparseresidency)
growth(dev, s::Committed, need) = need <= sizeof(s.region) ? InPlace() : outgrow(caps(dev), need, caps(dev).sparseresidency)
growth(dev, s::Reserve, need)   = need <= s.mapped ? InPlace() :
                                  need <= s.bytes  ? MapPages(s, min(s.bytes, max(need, 2 * s.mapped))) :
                                                     MoveTo(ReservePlan(reservesize(caps(dev), need)))
growth(dev, s::Evicted, need)   = RestoreTo(need)    # restored into storage of the new size (internals.jl section 9)
outgrow(c, need, ::HasSparse) = need >= sparsethreshold(c) && fitsreserve(c, need) ? MoveTo(ReservePlan(reservesize(c, need))) :
                                                             MoveTo(CommittedPlan(2 * need))
outgrow(c, need, ::NoSparse)  = MoveTo(CommittedPlan(2 * need))
# Core's policy from the device's facts; a device type reports the facts
# (c.sparsepage, c.sparseaddressspace, a total for the whole device) and
# decides nothing:
sparsethreshold(c; pages = 4) = pages * c.sparsepage   # smaller arrays double in the pool; the number is to be measured
reservesize(c, need) = min(nextpow(2, 4 * need), reservelimit(c))   # reservelimit: a share of c.sparseaddressspace (open)
# A need beyond reservelimit (lavapipe's 2 GiB of sparse address space) moves
# into Committed storage instead: a reserve is never smaller than the need.
fitsreserve(c, need) = need <= reservelimit(c)
# Acceleration structures never live in a reserve (VUID-03615): B.8.

# ── B.2 Array state ───────────────────────────────────────────────────────────

mutable struct ArrayState
    storage::Union{Region,Reserve,Committed,Evicted,Released}   # as applied so far: only an applying submission
                                                 #   changes it (B.4); Evicted under pressure (internals.jl section 9)
    dims::NTuple                                 # the requested dims (length(a)); the GPU's are in the cell
    resizable::Bool                              # set by the first resize! call (B.7); never cleared
    const lengthsource::Union{HostLength,DeviceLength}   # DeviceLength: created with capacity = n for a kernel-written length
    const capacity::Int                          # only for DeviceLength; 0 otherwise
    const cell::Cell                             # address and dims
    const pool::Pool                             # its device's pool: device(st), pool(st), nextseq!(st)
    const sync::SyncState                   # protects everything here, and the pending changes; has `freed` (B.10)
    const holds::Threads.Atomic{Int}             # the creator, each graph naming it, each composite holding it
    const composites::Threads.Atomic{Int}        # the holds of composites (a TLAS or BLAS range, an array GPURef)
    const dependents::Dict{WeakPlan,Dependence}  # every plan that names it (B.6)
    readers::ReaderSet                           # composites built from it (B.8); an immutable snapshot
end
# Readers (open decision 16, ruled B3.9/B3.10): the composites built from the array,
# counted per reader (a TLAS with two ranges over it counts twice; delete!
# removes one): a TLAS or BLAS, an array GPURef (which forwards to its own
# readers), a screen (a redraw mark). The snapshot is replaced under the
# array's lock, never changed in place, so a run iterates it without a lock
# (internals.jl pendreaders!) and a change can check under its locks that the
# snapshot it locked by is still current (B.4).
#
# Views (GPUArrays' derive) refer to this state with a static offset, dims and
# strides. Shrinking never unmaps an array's pages (they go back when the array
# is freed, or by trim!(a)), and no pressure step unmaps them, so a view never
# reads unmapped memory unless trim! was called.

# ── B.3 Changes: requests, scheduled ──────────────────────────────────────────

# A change is a step in the ordered stream, like any operation (Simon: "it
# should get scheduled"). The call only enqueues it: under the resource's
# lock it appends the request, with a device-wide sequence number, to the
# resource's pending list and updates what the host reports (length(a) is the
# requested length: a number; nothing physical changes). Nothing of the
# array is allocated, nothing is copied to the device, and it never starts
# over; a store's only work is one copy of its data into an upload region
# (below, B3.18). New storage, pages, the copy of the contents, the cell write and the
# retirement of the old storage are done by the submission that applies the
# change (B.4), in sequence order, after every earlier access.
abstract type Change end
# a[r] = data, copyto!, copy!: the data is copied once, at the call, from the
# caller's memory into an upload region: host-visible system memory acquired
# for this store before the lock (Simon, 2026-10-05, B3.18). The region is not
# the array's storage, so nothing about the array changes at the call; it is
# retired with the token of the submission that applies the store (or with
# none when a later store covering it replaces it, or the array is released).
# `span` is the byte range in the parent (through a view's static offset).
# Which storage it lands in is decided when it is applied: the storage the
# array has at that point of the sequence, so a store pended before a resize
# is copied along with the old contents, and one pended after it writes the
# new storage.
struct Store <: Change
    seq::UInt64; target::ArrayState; upload::Region; span::UnitRange{Int}
end
# resize!: the requested dims, and how many bytes of the old contents to keep
# (Keep: up to the smaller size; Discard, copy!'s case: none).
struct Resize <: Change
    seq::UInt64; target::ArrayState; dims::NTuple; keep::Int
end
struct Rebind <: Change                # r[] = b for an array GPURef: its cell gets b's cell index
    seq::UInt64; target::ArrayBox; new::MantleArray
end
struct BuildAccel <: Change            # a TLAS or BLAS: Build() or Update(); the capacity is the applier's (B.8)
    seq::UInt64; target::AccelStructure; mode::Union{Build,Update}
end
# The steps an applying submission emits for these (never pended themselves):
# BindPages (pages of a reserve, through the sparse queue), MoveStorage (a
# copy into new storage), CellWrite (address and size), MoveAccel (a
# structure's new storage and scratch), Restore (an evicted array's host copy
# back).
uses(c::Change) = ((c.target, Write()),)
# A Rebind reads its new target: the applying step waits for the target's
# earlier writes (eager ones, such as its creation upload, and the last runs
# of the plans writing it) and locks those plans (internals.jl lockstage,
# expanded).
uses(c::Rebind) = ((c.target, Write()), (c.new.state, Read()))
uses(c::BuildAccel) = ((c.target, Write()), ((m, Read()) for m in allmembers(c.target))...)   # transitively: a GPURef's target too

# Pending changes coalesce per target, not per record (a TLAS, its records and
# its range table share one record): a later Store covering an earlier one
# into the same target, consecutive Resizes into one (the last dims, the
# smallest keep), the BuildAccels of one structure into one (a Build if any of
# them is a Build, else an Update; B.8), consecutive Rebinds into the last. A
# merged change takes the latest sequence number of those it merged, so it is
# applied after everything pended before any of them. Nothing merges across a
# change a running plan holds for after its run (takepending): a store pended
# before a held Resize and one pended after it stay two stores. A hold drop due after a
# change is applied is not a change: it waits in the record's `deferred`
# list, so coalescing never drops it.
pendone!(c::Change) = coalesce!(syncstate(c.target).pending, c)   # under the target's lock
nextseq!(x) = Threads.atomic_add!(device(x).changeseq, UInt64(1))   # x: anything on a device (an array state, a structure, a GPURef)

# The one shape of every change: resize!, stores, rebinds, push! and delete!
# on a structure. One locked call over `uses`, expanded to the readers of the
# arrays it changes (their structures get pending builds in the same call)
# and to the members of the composites it changes; `apply` checks what must
# hold at the call (not freed, in bounds), updates what the host reports, and
# returns the requests. It cannot fail for lack of memory: it allocates
# nothing. A composite written this way is correct by construction: its
# consumers write `apply`, nothing else.
function change!(apply, uses)
    lockexpanded(uses; readers = true) do recs, x
        cs = apply()
        foreach(pendone!, cs)
        foreach(c -> notifyplans!(c.target), cs)     # into the inboxes of the plans naming each changed resource
        Pended()                                     #   (and naming composites built from it; internals.jl section 8)
    end
end

# ── B.4 Applying changes: what the submitter does ────────────────────────────

# Every submitter (a run's stage, an eager call, a host read, a vendor view,
# an eviction copy) applies the pending changes of what it locks, in front of
# its own work, in the same submission. Two steps:
#
# 1. Before any lock, prepare!(prepared, records): from the pending lists of
#    the records it will lock (read without their locks: a snapshot) and from
#    their storage, decide what applying them needs, and allocate it, one item
#    at a time into `prepared`: for a Resize, growth(dev, storage at that
#    point, requested size) (B.5) gives InPlace, Adopt, pages to map, or new
#    storage; for an evicted array (or a composite member evicted before it
#    became one), restore storage of its requested size; for a BuildAccel of a
#    structure whose total outgrew its capacity, the doubled AS storage,
#    scratch, records and range-table storage (B.8). This is where pressure
#    happens (acquire!: unmapping, eviction, waiting for retirement), and
#    the only place a change allocates storage (a Store's upload region was
#    acquired at its call, B.3).
# 2. Under the locks, submitwith! takes the pending changes in sequence order
#    and emits them with the prepared items (take!). If an item is missing or
#    no longer fits (a change arrived, was coalesced or was applied between the
#    two steps), it returns Again before emitting anything; the submitter
#    releases what it prepared and starts over from step 1. Whatever was
#    prepared and not taken goes back.
#
# The stale mark for plans that compile decided something about a length
# (a direct launch over it, a temporary sized for it, a TLAS total) is set at
# the call (marklength!, marktotal!): a flag, the next run recompiles, nothing
# physical happens. Everything else about staleness is decided when the
# change is applied:
#   - a move of storage a plan recorded by handle marks the parts of that plan
#     that recorded it (open decision 17 (c): they are recorded again before
#     they are next submitted, after the prefix that moved the storage, like
#     a marked AccelCommand); nothing recompiles. A plan between its stages
#     gets the same: its later stage re-records, and the move, ordered after
#     its earlier stages' accesses, has copied what they wrote;
#   - a change the applying plan cannot follow mid-run (a direct launch over
#     the array's length, a temporary sized for fewer elements: compile
#     decided those) is left pending by that plan's own later stages, with
#     every later change on the same target (it is applied after the run, and
#     the next run recompiles), and makes any other submitter return WaitFor:
#     it waits outside its locks until that run has submitted its last stage,
#     and starts over (from inside a host function it throws: it could wait
#     in a cycle; inhostfunction below; another run past its stage 1 leaves
#     the change for after its own run instead). Plans not between their
#     stages are only marked stale (B.6).
# A run past its stage 1 never waits: a change another plan's run cannot
# follow is left, with every later change on its target, for after this run,
# like the changes it cannot follow itself. Every other submitter (stage 1 of
# a run included) gets WaitFor and waits outside its locks: a run's stage 1
# outside its graph lock too (internals.jl run!(g)).
function takepending(records, self)
    cs = sort!([c for r in records for c in r.pending]; by = c -> c.seq)
    held = heldforrun(cs, self)                      # changes self's later stages leave for after the run
    w = waitingfor(setdiff(cs, held), self)          # other plans between stages that a change would stale
    if w !== nothing
        pastfirst(self) || return w                  # WaitFor
        held = [held; blockedfrom(setdiff(cs, held), w)]   # those changes and the later ones on their targets
    end
    taken = setdiff(cs, held)
    return Taken(taken, unique(syncstate(c.target) for c in taken))
end
pastfirst(::Nothing) = false                         # not a run
pastfirst(pl::Plan) = between(pl)                    # set once stage 1 has submitted

# The full submit path, shared by all submitters (under the locks of
# `records`, after step 1). The submitter's own work is recorded already:
# `ownparts` are recorded parts and AccelCommands. The changes go into a small
# command buffer in front of it, in the same submission. Only after the
# changes are emitted are the parts' command buffers taken: a marked
# AccelCommand, or a part that recorded a handle a move just replaced, is
# recorded again then (B.8, B.6). The changes' uses are added to the
# submitter's: the submission waits for every earlier access the changes
# conflict with, on any queue, and records the changes' writes as the
# resources' last writes, so later readers on other queues wait for them, and
# storage a move replaced is retired with this token only after every earlier
# reader was waited for. `ownwaits` lets a run pass the waits it computed for
# all its segments before recording any of them. Returns Again or WaitFor
# (nothing emitted), or the token and the records whose changes it applied (a
# later segment of a run that touches one of them waits on this token;
# internals.jl section 6).
function submitwith!(ch, records, prepared, ownuses, ownparts; self = nothing, restores = true,
                     ownwaits = waitsfor(ownuses), present = nothing)
    taken = takepending(records, self)
    done(taken) || return taken                      # WaitFor
    steps = Step[]
    for r in records                                 # evicted resources first (internals.jl section 9); a run
        restores || break                            #   looks only when its plan is marked for restore
        s = restoresteps(state(r), state(r).storage, takeor(prepared, state(r)))
        done(s) || return Again()
        append!(steps, s)
    end
    for c in taken.changes
        s = stepsof(c, prepared)                     # B.5, B.8: the steps, with the prepared items taken
        done(s) || return Again()                    # missing or too small: nothing emitted, start over
        append!(steps, s)
    end
    binds = Token[]
    foreach(s -> bindnow!(ch, s, binds), steps)      # pages first, through the sparse queue
    pre = Builder(device(ch), ch)
    foreach(s -> emitstep!(pre, s), steps)           # copies, stores, cell writes, record writes, builds, with barriers
    cmds = recordown(ownparts, steps)                # internals.jl section 6; an eager call packs here
    changeuses = [u for c in taken.changes for u in uses(c)]
    # The steps' accesses, and a non-run submitter's own (`ownuses`; a run
    # passes none: its accesses are its segments' tokens), are recorded as
    # eager accesses on the SyncStates and put into the inboxes of the other
    # plans naming those resources (internals.jl section 8). Their waits cover
    # the naming plans' last runs, read under those plans' sync locks.
    token = submit!(ch, [finish(pre); cmds...]; eager = [changeuses; ownuses...], self,
                    waits = [waitsfor([changeuses; ownuses...]); ownwaits; binds], present)
    commit!(taken, steps, token)   # remove from the records; update the storage they changed; retire
    return token, taken.records    #   replaced storage with token; deferred hold drops with (token,)
end
# A change of a structure pends the Updates of the structures over it (its
# readers) in the same change! call, which locks them (readers = true), with
# later sequence numbers: a BLAS Build is followed by its TLASes' Updates, in
# one submission if it takes both, and otherwise the TLAS's Update waits, when
# some submission takes it, for the BLAS's write through the sync states
# (a build reads its members).
# Pages are bound on the sparse channel through sparsesubmit! (internals.jl
# section 9, the device type's bindpages!), marked bound so a retried
# submission does not bind them again, and the submission waits on the tokens.
bindnow!(ch, s::Step, binds) = nothing
bindnow!(ch, s::BindPages, binds) = s.bound === nothing &&
    push!(binds, (s.bound = sparsesubmit!(sparsechannel(ch), bindpages!, (s.reserve, s.pages), ((s.target, Write()),))))

# Plans a change makes stale and that are between their stages (other than
# `self`): the plans among the dependents (of the array, or of the GPURef for
# plans that bound its target by handle, B.6) for which the change is one
# compile decided (stales(d, c)); handle dependences re-record instead (above).
# `between` is set and cleared under the record locks a run takes (internals.jl
# section 6): a plan not yet past its stage 1 sees the stale mark and starts
# over, so a submitter never waits on a run that is itself waiting.
function waitingfor(cs, self)
    w = Plan[]
    for c in cs, (wp, d) in alldependents(c.target)
        p = wp.plan.value
        p === nothing || p === self || !between(p) || !stales(d, c) || push!(w, p)
    end
    isempty(w) ? nothing : WaitFor(unique(w))
end
alldependents(r::ArrayBox) = r.dependents
alldependents(t::AccelStructure) = t.dependents  # plans sized from it (Raycore's build temporaries, D.1)
alldependents(st::ArrayState) = Iterators.flatten((st.dependents, (e for r in st.readers for e in boundthrough(r))))
boundthrough(r::ArrayBox) = r.dependents      # plans that bound its target by handle
boundthrough(_) = ()                          # a TLAS, a BLAS, a screen
# Outside every lock. A task inside a host function of a running graph
# throws before waiting on anything: it could wait for its own run. Not
# detected: a task a host function starts and waits for; like two host
# functions that run each other's graphs, that can deadlock (internals.jl
# lockgraph). No fairness: the submitter starts over when the run ends and
# may wait again if the plan's next run got past its first stage first.
function waitforruns(w::WaitFor)
    inhostfunction(current_task()) && throw(ArgumentError(
        "used an array a running graph cannot follow mid-run (a direct launch over its length, a temporary sized for it); use it before or after the run"))
    for p in w.plans
        lock(() -> while between(p); wait(p.runend) end, p.runend)
    end
end
waitforruns(::Again) = nothing
# Whether the task is running a host function of some graph (runhost! sets
# g.hostcaller around the call). A declaration or compile holding a graph's
# lock is not "inside a run". A scan of the device's graphs, held weakly;
# current_task() is used only for this check, as in lockgraph.
inhostfunction(task) = any(g -> (@atomic g.hostcaller) === task, livegraphs(REGISTRY))

# Composite resources: a TLAS reaches its range sources and BLASes through
# cells, a BLAS its geometry, an array GPURef its target, so no plan names
# the members. Every submitter locks with lockexpanded instead of locked: it
# reads each composite's member snapshot (immutable, replaced on every change)
# transitively (TLAS → BLASes and range arrays, BLAS → geometry, GPURef →
# target), locks the union in id order, checks under the locks that every
# snapshot it read is still current (===), and starts over otherwise. A record
# a composite owns brings the composite's members too: a TLAS's records and
# range table share its record, so Array(t.records), which locks that record
# and so takes t's pending build, has t's members locked as well (their
# pending stores are applied first, and the build's reads of them are
# recorded under their locks). A change (change!, readers = true) also
# expands an array to its readers' snapshot, transitively (array → GPURef →
# TLAS), since it pends their builds.
# `f` gets the records and the expansion `x`; expand(x, uses) rewrites the
# uses of composites, transitively:
#   a read of a composite         → a read of it and of each member;
#   a build or refit (BuildUse)   → a write of the structure, a read of each member;
#   a write through an array GPURef → a read of the GPURef, a write of its target;
#   AnyWrite / AnyMemory (compile's placeholders for unknown members) → dropped:
#     the expanded uses name the members.
# Submitters that are not runs (eager calls, host reads, vendor views,
# eviction copies, and the steps a run applies) wait for and record the
# expanded list. Runs use neither: between graphs, a plan naming a composite
# is among its members' namingplans (through their readers snapshots), so a
# simulation step writing an array a frame draws through a GPURef waits for
# the frame's last run and puts the array into the frame's inbox, and the
# frame's next run waits for the step (internals.jl section 8). Runs expand
# only what their inbox names (internals.jl lockstage).
function lockexpanded(f, uses; readers = false)
    while true
        x = expansion(uses; readers)                  # each snapshot read under its own lock, briefly
        r = locked(records(uses, x)) do               # the uses' records and every member's, by id
            current(x) || return Again()
            Done(f(records(uses, x), x))
        end
        done(r) && return r.value
    end
end

# ── B.5 resize! and stores ────────────────────────────────────────────────────

Base.resize!(a::MantleArray, dims) = resize!(a, dims, a.role)
Base.resize!(a, dims, ::Derived) = throw(ArgumentError("resize! of a view; resize its parent"))
Base.resize!(a::MantleArray, n::Integer) = resize!(a, growlast(size(a), n))   # sets the last dimension
Base.resize!(a::MantleArray{T,N,<:Device}, dims, ::Root) where {T,N} = resizeto!(a, dims, Keep())

# The request, and nothing else. The length change reaches the array's
# readers in the same locked call (their structures get a pending Build; the
# capacity a new total needs is the applier's, B.8). Plans with a direct
# launch over the array's length or a temporary sized for fewer elements are
# marked stale here (that depends only on the requested length, B.6); whether
# the storage moves is decided when it is applied.
function resizeto!(a::MantleArray, dims, contents)
    st = a.state
    change!(((st, Write()),)) do
        usable(a)                                     # freed through the creator's handle: throws, under the lock
        keep = keepbytes(contents, st, dims)          # Keep: the smaller of the requested sizes so far and now
        st.dims = dims                                # what length(a) reports
        st.resizable = true
        marklength!(st, dims)                         # B.6
        Change[Resize(nextseq!(st), st, dims, keep); (lengthchange(t, st) for t in st.readers)...]
    end
    return a
end
# A graph array: the same within the size its slot was placed for; beyond it,
# the graph recompiles at its next run (under the graph lock).

# Applying a Resize (step 1 of B.4 decided and allocated, step 2 emits), by
# dispatch on (storage at that point, prepared growth):
stepsof(c::Resize, p) = resizesteps(c.target, storageat(c, p), takeor(p, c), c)   # storageat: after the earlier steps
resizesteps(st, s, ::Missing, c) = Again()            # not prepared: start over
resizesteps(st, s::Union{Region,Reserve,Committed}, ::InPlace, c) = (CellWrite(st, address(s), c.dims),)
resizesteps(st, r::Region, ::Adopt, c) = (Becomes(st, Committed(r)), CellWrite(st, address(r), c.dims))
resizesteps(st, s::Reserve, g::MappedPages, c) = (BindPages(st, s, g, nothing), CellWrite(st, address(s), c.dims))
resizesteps(st, old, new::Reserve, c) =               # a move into a new reserve binds its pages first
    (BindPages(st, new, firstpages(new), nothing), MoveStorage(st, old, new, min(c.keep, sizeof(old))),
     CellWrite(st, address(new), c.dims))
resizesteps(st, old, new::Committed, c) =
    (MoveStorage(st, old, new, min(c.keep, sizeof(old))), CellWrite(st, address(new), c.dims))
resizesteps(st, ev::Evicted, new::RestoreStorage, c) =   # restored into storage of the requested size
    (Restore(st, ev.host, new.storage, min(c.keep, sizeof(ev.host))), CellWrite(st, address(new.storage), c.dims))
# A move is a copy, a write of the array: applied, it waits for every earlier
# access on any queue (B.4), and the old storage is retired with its token. A
# step whose storage no longer fits (prepared for a size another change
# replaced) is Again. A move of storage a plan recorded by handle marks that
# plan's parts for recording again (B.4, B.6).

# Stores: the data is copied once at the call, into the store's upload
# region; the rest is the applier's: a small store is written inline by the
# applying submission from the region's bytes (emitstore!: vkCmdUpdateBuffer
# up to the device type's limit, caps facts c.inlinestore and c.inlinealign;
# Vulkan: 65536 bytes at 4-byte offsets and sizes,
# chapters/clears.adoc:800-805), a large one by a copy from the region. A store into an array with readers pends their Update
# (a store into a TLAS source is a refit; into a BLAS's index array a Build,
# B.8). A store into an evicted array is applied after its restore, which the
# applier does first (B.4 step 1). A store through a view with non-unit
# strides is not one byte range: the packed elements go into the upload
# region like any store, and the applying submission scatters them (a copy
# kernel in its prefix, in sequence like any step). The only wait a store
# can make is acquiring its upload region under memory pressure (acquire!,
# internals.jl section 9), before any lock.
function Base.setindex!(a::MantleArray{T,N,<:Device}, data, r::AbstractRange) where {T,N}
    length(data) == length(r) || throw(DimensionMismatch("setindex! of $(length(data)) elements into $(length(r))"))
    st = a.state
    span = parentbytes(a, r)
    up = acquireupload!(pool(st), length(span))       # host-visible system memory, before any lock
    copyto!(hostview(up, T), vec(data))               # the one copy, from the caller's memory
    ok = false
    try
        change!(((st, Write()),)) do
            usable(a)                                 # freed through the creator's handle: throws, under the lock
            checkbounds(a, r)                         # against the requested size
            Change[Store(nextseq!(st), st, up, span); (update(t, st) for t in st.readers)...]
        end
        ok = true
    finally
        ok || retire!(pool(st), up, ())               # a throw under the lock: the region goes back
    end
    return a
end
stepsof(c::Store, p) = storesteps(c, storageat(c, p), inlineor(c, caps(device(c.target))))
storesteps(c, s, ::Inline) = (InlineStore(c.target, s, c.span, c.upload),)   # emitstore! from the region's bytes
storesteps(c, s, ::Staged) = (StagedStore(c.target, s, c.span, c.upload),)   # a copy from the region
# inlineor: Inline() within c.inlinestore at c.inlinealign offsets and sizes, else Staged();
# takeor(p, key): the item prepared for key, or Missing(). It is only taken
# (kept from release!) when commit! runs: an Again or a throw before the
# submission leaves every item to the submitter's release!.
Base.copyto!(a::MantleArray{T,1,<:Device}, data::AbstractVector) where {T} = (a[1:length(data)] = data; a)
# copy! with a new length: the Resize (old contents not kept) and the Store,
# one after the other in the sequence, pended in one locked call so nothing
# can come between them.
Base.copy!(a::MantleArray{T,1,<:Device}, data::AbstractVector) where {T} = copy!(a, data, a.role)
# The same length is a plain store: nothing about the array's size changes,
# so no Resize, no `resizable`, and no stale mark for plans that launch
# directly over it (RayMakie's setattribute! calls copy! on every update).
function copy!(a, data, ::Root)
    st = a.state
    n = length(data)
    up = acquireupload!(pool(st), sizeof(eltype(a)) * n)   # before any lock (B3.18)
    copyto!(hostview(up, eltype(a)), vec(data))      # the one copy
    ok = false
    try
        change!(((st, Write()),)) do
            usable(a)
            span = parentbytes(a, 1:n)
            n == length(a) && return Change[Store(nextseq!(st), st, up, span); (update(t, st) for t in st.readers)...]
            dims = (n,)
            st.dims = dims
            st.resizable = true
            marklength!(st, dims)
            Change[Resize(nextseq!(st), st, dims, 0); Store(nextseq!(st), st, up, span);
                   (lengthchange(t, st) for t in st.readers)...]
        end
        ok = true
    finally
        ok || retire!(pool(st), up, ())
    end
    return a
end
copy!(a, data, ::Derived) = length(a) == length(data) ? (a[1:length(data)] = data; a) :
    throw(ArgumentError("copy! of $(length(data)) elements into a view of $(length(a)); resize its parent"))

# ── B.6 Dependents ────────────────────────────────────────────────────────────

# Every plan that names an array registers in its dependents, so an eviction
# can mark the plans that must restore it before their next run (internals.jl
# section 9). A plan is made stale only where compile recorded something the
# cell cannot update:
struct Dependence
    direct::Bool        # a direct launch or draw with its length (the array was fixed at compile)
    handle::Bool        # its buffer handle in a command: an index-buffer bind, a build input, a
                        #   Vulkan buffer copy (open decision 17: re-record instead of recompile)
    sizedfor::Int       # a temporary or a selection sized for this many elements (typemax if none)
    compiled::Tuple     # the storage and length compile saw
end

# Marked where it is known, under the array's lock:
#   at the call (marklength!): a requested length marks plans sized for fewer
#   elements and plans with a direct count (the first resize! of a fixed
#   array, which compile launched over directly);
#   when a move is applied (B.4): plans that recorded the storage's handle get
#   the parts that recorded it marked for recording again (open decision 17
#   (c)), not a recompile; the stale flag is not set for them.
# Marking stale sets the plan's stale flag and the graph's recompile flag,
# through weak references; a collected plan or graph is skipped. Compile reads
# the requested state: an array is resizable from its first resize! call (the
# `resizable` flag, a host fact set at the call), whether or not the Resize
# has been applied, so a plan compiled after the call launches over it
# indirectly; a handle compile records is the storage applied so far, and a
# pending move re-records it when applied.
function markstale!(w::WeakPlan)
    pl = w.plan.value; pl === nothing && return
    @atomic pl.stale = true
    g = pl.graph.value; g === nothing || (@atomic g.recompile = true)
end

# Registration, after compile and record, before the plan becomes the graph's
# plan: under each dependency's lock, compare what compile saw with the state
# now (storage identity, length, dims against sizedfor); if anything moved in
# between, return Again and compile again. The structures the plan builds or
# refits are locked too: their command lists change here, and mark! walks
# them under the same lock. A plan unregisters itself when it is retired
# (recompile!, release!), under the same locks, so the tables do not grow and
# a retired plan is never marked again.
# A dependence's `state` is an array state or, for a target bound by handle
# through an array GPURef, the GPURef: matches compares its current target
# with the one compile resolved, and registering takes the plan's own hold on
# that target (it cannot fail: the GPURef holds it, and its lock is held
# here). The plan drops that hold when it retires (unregister!).
# Registering adds the plan to the `namers` of every resource it names, with
# its use from compile (internals.jl section 8; the members of the composites
# it names reach it through their readers snapshots), and puts the resources
# it reads or writes that another plan names, or that have eager accesses,
# members included, into its own inbox: its first run waits for their other
# users like any later run. The sync locks of the plans naming those
# resources come first (lock order), from a snapshot checked under them; a
# plan registering or retiring meanwhile only makes the locking start over,
# nothing recompiles. This is the one place that walks a composite's members
# for ordering, at compile, not at run. unregister! does the reverse under
# the same locks: it leaves the namers, adds its last-run tokens to the
# `retired` list of every resource it named (members of the composites it
# named included), and puts those another plan names into that plan's inbox,
# so their next accesses wait for its last run and storage it used is
# released only after it.
function register!(pl::Plan)
    while true
        xs = expandeduses(pl)                        # what it names, and the members of its composites as reads
        others = otherplans(pl, xs)                  # a snapshot: the plans naming any of them
        r = lockplans([pl; others]) do
          locked([dependencyrecords(pl); [syncstate(c.structure) for c in accelcommands(pl)];
                  [syncstate(r) for (r, _) in xs]]) do
            expandeduses(pl) == xs || return Retry()           # a member was added meanwhile
            otherplans(pl, xs) ⊆ others || return Retry()      # lock again with the new plan
            all(d -> matches(d.state, d.dependence.compiled), pl.depends) || return Again()   # stale: recompile
            foreach(((r, u),) -> addnamer!(syncstate(r), pl, u), nameduses(pl))   # direct names; snapshots replaced
            foreach(((r, _),) -> shared(r, pl) && pushinbox!(pl, r), xs)
            foreach(d -> (d.state.dependents[WeakPlan(pl)] = d.dependence; holdhandletarget!(pl, d.state)), pl.depends)
            foreach(c -> push!(c.structure.commands, c), accelcommands(pl))   # marked by later changes (B.8)
            foreach(r -> lock(() -> push!(r.plans, WeakPlan(pl)), r.lock), valuerefs(pl))   # entry marks (internals.jl section 7)
            Registered()
          end
        end
        retried(r) || return r                       # Registered or Again
    end
end
retried(::Retry) = true
retried(_) = false
# Also a resource with pending changes: a store or a new TLAS's zero-instance
# Build pended before this plan existed reached only the plans naming it then.
shared(r, pl) = any(q -> q !== pl, namingplans(r)) || !isempty(eageraccesses(syncstate(r))) ||
                !isempty(syncstate(r).pending)
# A change of a composite's members (push!, delete!, a rebind) changes no
# plan. The member's readers snapshot is replaced at the call (B.8, B.8b), so
# from then on the plans naming the composite are among the member's
# namingplans: the member's writers lock them and wait for their last runs.
# Those plans read the new member only after the step applying the
# composite's Build, which holds the sync locks of the member's writers
# (internals.jl lockstage, expanded). A removed member keeps the composite as
# a reader until the submission that removed it has passed (delete!, B.8).

# A run checks the stale flag under its records' locks before it claims a
# token (the mark is set under an array lock the run also takes) and returns
# Again; run!(g) recompiles. A change a running plan cannot follow mid-run is
# left for after the run by the plan's own later stages and makes every other
# submitter wait until the run has submitted its last stage (B.4,
# takepending): the call that pended it never waits. A plan that registers,
# or starts running, after the change was pended finds its stale flag set
# under the locks and recompiles first.
#
# A plan that binds a GPURef's target by handle (an index buffer through an
# array GPURef) depends on the GPURef (its `dependents`, not the target's):
# compile records the target it resolved, register! checks under the
# GPURef's lock that it is still the target and takes the plan's hold on it,
# held until the plan retires. A rebind between compile and register! makes
# the plan compile again; a rebind applied after it marks the parts that
# recorded the old target's buffer for recording again (open decision 17
# (c)); the old target is never freed under a plan that recorded its buffer. Plans that bind an array
# directly are not marked by a rebind of some GPURef that also points at it.
holdhandletarget!(pl, r::ArrayBox) = addhold!(r.target.state)   # dropped by unregister!
holdhandletarget!(pl, st) = nothing

# ── B.7 Lengths at compile ────────────────────────────────────────────────────

# launchrange(a) puts a reference to the array into the node, not an Int; the
# decision is made at compile from the storage then.
launchrange(a::MantleArray) = LengthOf(a)
extents(s, r::LengthOf) = extents(s, r, fixedness(r.array.state), r.array.state.lengthsource)
extents(s, r::LengthOf, ::Fixed, ::HostLength) = (depend!(s, r.array; direct = true); Direct(length(r.array)))
extents(s, r::LengthOf, _, _) = Indirect(Bound(r.array))       # the head (or a prepare node) writes the command
fixedness(st) = st.resizable ? Resizable() : Fixed()          # (a field typed by state in real code)

# Temporaries sized from an array: a size expression evaluated at compile.
# Host length: twice the current length. Kernel-written length: the capacity.
MantleArray(l::GraphDevice, T, s::SizedBy) = transient(l, T, s)   # s = SizedBy(nblocks, a)
MantleArray(dev::Device, T, s::SizedBy) = MantleArray(dev, T, s.f(sizing(s.array, s.array.state.lengthsource)))

# An array created from data uploads now (B3.18): creation is not a change,
# since no graph can name the array before it exists. The storage is
# allocated, then the data is streamed from the caller's memory (or a mapped
# file, MantleArray(path; mmap = true), B3.9) in chunks, each copied into an
# upload region and written by one eager submission cut to the submission
# budget, or written directly where the storage is host-visible. At most a
# few chunks are in flight, so the host memory it needs is bounded by them,
# not by the array. It returns once every chunk is submitted; the writes are
# recorded on the array's sync state, so later uses wait for them.
function MantleArray(dev::Device, data::AbstractArray{T,N}) where {T,N}
    a = MantleArray(dev, T, size(data)...)
    for chunk in uploadchunks(dev, sizeof(data))      # byte ranges sized from caps(dev) and the budget
        writechunk!(a, data, chunk)                   # host-visible storage: a direct copy; else upload region +
    end                                               #   submitapplying (internals.jl section 11), region retired with its token
    return a
end
placesize(s, x::SizedBy) = (n = sizing(x.array, x.array.state.lengthsource); depend!(s, x.array; sizedfor = n); x.f(n))
sizing(a, ::HostLength) = 2 * length(a)
sizing(a, ::DeviceLength) = a.state.capacity

# Eager launches over a kernel-written length are indirect from the cell too:
# one behaviour for the eager and the graph form.

# Draws: vertex, index and instance counts are numbers or LengthOf
# references; indirect draws from Bound slots unless the array is fixed, and
# always inside a when! block (the count is multiplied by the flag;
# internals.jl section 2). A count of 0 draws nothing. Vertices are pulled by
# address. An index buffer is bound by handle (a dependence with handle =
# true; open decision 17).

# Graph arrays: resize! within the size the slot was placed for is a pending
# cell write; beyond it the graph recompiles at its next run (graph lock).
# Window-sized arrays are placed for the window's capacity (the largest
# monitor unless given); a window larger than that recompiles the graph too.
# Window-sized images are created per swapchain generation at the window's
# size on the same arena memory, which maps only the pages that size needs
# (internals.jl section 7).

# ── B.8 Acceleration structures ───────────────────────────────────────────────

# One core type per level; a device type builds hardware structures, Raycore
# software ones. Both levels are composites (B.4): they read their members
# through cells, hold them (composite holds, internals.jl section 10) and are
# their readers (B.2).
abstract type AccelStructure end
mutable struct TLAS{P} <: AccelStructure     # P: Hardware() or Software()
    const device::Device
    const path::P
    slots::Vector{Union{Nothing,InstanceRange}}  # host side; a deleted range's slot is reused
    members::MemberSet                       # immutable snapshot: sources, transform inputs, BLASes
    const rangetable::MantleArray{RangeEntry,1}   # on the device; shares the TLAS's sync state
    const records::MantleArray               # instance records (device type's record type); shares it too
    const offsets::MantleArray{UInt32,1}     # range offsets, read by shaders for per-instance data
    const kernels::RecordKernels             # the offsets and records kernels (below)
    capacity::Int                            # instances the storage and scratch are sized for; builds use the count
    builtcount::Int                          # the count of the last Build, -1 before the first; every Update carries it
    appliedtotal::Int                        # the total as of the changes applied so far (commit!); what commands record
    totalsource::Union{HostTotal,DeviceTotal} # DeviceTotal once a range's source has a kernel-written length (host view,
                                             #   set by push!/delete!, which mark!); then builds are over the capacity
    storage::AccelStorage                    # hardware: AS object on a pool Region; software: Raycore's arrays
    scratch::Region                          # build scratch for `capacity`, replaced with it (not a graph transient)
    const commands::WeakSet{AccelCommand}    # plans' builds and refits of it, marked by changes (below)
    const dependents::Dict{WeakPlan,Dependence}   # plans sized from its capacity (D.1's temporaries; B.6)
    const cell::Cell                         # AS address (Vulkan) or resource ID (Metal)
    const countcell::Cell                    # the total, written by the offsets kernel; read where the device reads counts from memory
    const sync::SyncState
    const holds::Threads.Atomic{Int}
end
members(t::TLAS) = t.members.resources
# The TLAS holds its own arrays (range table, records, offsets; the software
# storage arrays) with composite holds, so none of them is ever evicted
# (internals.jl section 9) and they are released with it. They are created
# resizable (Committed, B.1): launches over them, Raycore's build kernels
# among them, are indirect, so their growth with the capacity makes no plan
# stale and never waits for a run (B.3).

struct InstanceRange
    blas::BLAS
    source::Union{MantleArray,ArrayBox}      # transforms or positions; a single transform is a 1-element array
    inputs::Tuple                            # scale, rotation: arrays or one value
end
# On the device a range names cells, not addresses: every array it reads has a
# cell index (an array GPURef's cell holds its target's index). Slots are
# stable for a range's life: instanceCustomIndex is the slot, and shaders find
# an instance's index within its range as InstanceId - offsets[slot].
struct RangeEntry
    blascell::UInt32
    sourcecell::UInt32
    inputcells::NTuple{2,UInt32}             # scale, rotation; 0 for "one value", in valueslots
    valueslots::NTuple{2,Float32x4}
    mask::UInt8                              # the instances' visibility mask: 0 hides them, a refit
    active::Bool                             # false only for a deleted range's slot: inactive records, a Build
end

# Records: two kernels compiled when the TLAS is created (t.kernels; fixed
# arguments: the range table, the offsets and the records, read through
# their cells, so a move of the records does not stale them, and the cell
# arena). One computes each slot's offset from the lengths in the source cells
# (a prefix sum) and writes the total into countcell; one writes every
# instance record (transform from the source and inputs, mask from the entry,
# BLAS address or resource ID from the BLAS cell), and records with mask 0
# from the device total up to the build count, so a count that ran ahead of
# a source's cell builds no unwritten record. Both kernels read cells
# directly (about 195-220 ns more than a slot read, once per build).
#
# Capacity is the applier's, like an array's storage: push!, delete! and a
# source's resize! only pend a Build (and the range-table entry). The
# submitter that applies the Build computes the total from the requested
# state; if it outgrew the capacity, step 1 of B.4 allocated the doubled AS
# storage, scratch, records and range-table storage (preparecapacity), and
# the Build's steps start with the capacity steps (B.8 capacitysteps). Missing
# or too small: Again, nothing emitted.
#
# The change prefix emits the BuildAccel changes of one structure coalesced
# into one (B.3): a Build if any of them is a Build. A Build records its count
# as builtcount and, when that count is new, marks the structure's commands,
# so every Update command is recorded again with it (B.4: after this prefix).
# An Update carries builtcount, never the current total, since an update must
# keep the count of the last build (header); an Update before any Build is a
# Build. The first prepare kernel's barrier is a global memory barrier after
# everything the prefix wrote before it (cells, the range table, stored
# source data); each later step has its own. The prepare kernels are the
# structure's: the offsets and records kernels of a TLAS, the zero-area fill
# of a BLAS over kernel-written lengths (none for other BLASes), the same
# list current! records.
stepsof(c::BuildAccel, p) = buildsteps(c, outgrown(c.target), takeor(p, c))
buildsteps(c, ::Fits, _) = (BuildStep(c.target, effectivemode(c.target, c.mode)),)
buildsteps(c, ::Outgrown, ::Missing) = Again()        # not prepared: start over
buildsteps(c, ::Outgrown, grown) = (capacitysteps(c.target, grown)..., BuildStep(c.target, Build()))   # a new AS: a Build
function emitstep!(b, st::BuildStep)
    s = st.structure
    after = emitprepared!(b, preparekernels(s), Deps((), barrier(PrefixWrites() => AnyRead())))
    emitbuild!(b, nothing, compiledbuild(s, st.mode, buildcount!(s, st.mode)), after)
end
# emitprepared!(b, kernels, first): the first kernel with `first`, each later
# one with a barrier after the one before it (the records kernel reads the
# offsets the offsets kernel wrote). Returns the build's Deps: the barrier
# after the last kernel, or `first` itself merged into the build's barrier
# when there is no kernel (a BLAS with no kernel-written length), so the
# barrier is never lost.
function emitprepared!(b, kernels, first)
    deps = first
    for k in kernels
        emitdispatch!(b, nothing, k, deps)
        deps = Deps((), barrier(Write(k) => Read(Next())))
    end
    return merge(deps, Deps((), barrier(PrepareWrites() => BuildInput())))
end
effectivemode(t, ::Build) = Build()
effectivemode(t, ::Update) = t.builtcount < 0 ? Build() : Update()   # nothing built yet: a refit is a Build
# Under t's lock, held by the submitter. A new count marks t's commands, so
# every Update command is recorded again with it after this prefix (B.4).
function buildcount!(t, ::Build)
    n = hostcount(buildcount(t))
    n == t.builtcount || (t.builtcount = n; mark!(t))
    return n
end
buildcount!(t, ::Update) = t.builtcount

# The count a build or refit carries, by device type. Vulkan's direct build
# takes it as a host value the driver records into the command. Vulkan's
# indirect build reads it from memory but is not used, on any driver: RADV
# 26.2.3 and NVIDIA 595 do not have it (accelerationStructureIndirectBuild =
# false), and on AMD's Windows driver, which has it, it is slower than the
# exact count because the driver builds for the maximum (N=100000: 1.08 ms
# exact; 1.17, 1.25, 1.94 ms with maximum 2N, 4N, 16N; m1_lapwin.txt). Metal
# reads it from a buffer (instanceCountBuffer). Storage sized for `capacity`
# takes any count up to it
# (vkdocs commonvalidity/build_acceleration_structure_nonindirect_common.adoc:13-17).
buildcount(t::TLAS) = buildcount(t, countsource(t.device))
buildcount(t::TLAS, ::CountFromMemory) = fromcell(t, t.totalsource)   # Metal
fromcell(t, ::HostTotal) = FromCell(t.countcell)
fromcell(t, ::DeviceTotal) = t.capacity      # built over the capacity on every device type, like Vulkan below
buildcount(t::TLAS, ::CountRecorded) = recordedtotal(t, t.totalsource)   # Vulkan, below
recordedtotal(t, ::HostTotal) = total(t)          # the host view, exact, no padding
recordedtotal(t, ::DeviceTotal) = t.capacity      # a kernel writes the total: built over the capacity
buildcount(b::BLAS) = primitives(b)    # the primitive count, host view (the geometry's capacity for a
                                       #   kernel-written length, below); a BLAS is never counted from memory
# hostcount(FromCell(_)) is the host view total(t) too: what builtcount keeps.
# A total a kernel writes has no host value. Such a TLAS (any device type) is
# built over `capacity`, with mask-0 records past the device total that
# reference a valid BLAS (a reference of 0 would make them inactive): the
# count stays the capacity and no record changes between active and
# inactive, so a refit after the kernel changed the total is legal (the
# padding measured in the header; the only padded case). A BLAS over arrays
# whose length a kernel writes is built over the geometry's capacity
# (maxVertex too, which an update must keep), with zero-area primitives past
# the device length written by a prepare kernel in its AccelCommand (zero
# area, not NaN: NaN would make them inactive). An update may change only
# instance definitions, transforms and vertex positions
# (chapters/accelstructures.adoc:159-170), so a run that writes a BLAS's
# vertices pends an Update and one that writes its index array a Build.
# (Both follow from B3.2; listed for confirmation in decisions.md.)

# Graph operations. A plan's build or refit of a TLAS or BLAS is an
# AccelCommand: its own small command buffer, submitted inside the segment's
# submission between the plan's command buffers. Its barrier (the node's,
# from barriers!: a full memory barrier, since the members are not known at
# compile; internals.jl section 4) is recorded with it, first. It records the
# count (Vulkan) and the storage handle by value. Nothing is compared at run:
# a change that alters what it recorded (a new count from push!, delete! or a
# source's length; a MoveAccel; a Build that sets a new builtcount) marks the
# structure's commands under the structure's lock, and the run records the
# marked ones again, after the change prefix (B.4). So a change of the count
# or the capacity never recompiles or re-records the plan. On Metal, which
# reads the count from the cell, a new count only matters to Updates.
build!(g::Graph, t::AccelStructure) = declare!(g, BuildNode(t, Build()))
refit!(g::Graph, t::AccelStructure) = declare!(g, BuildNode(t, Update()))
# refit!(g, t) runs the records kernels and the Update (they are part of the
# node); a refit before any build of t is a Build. A kernel-written total
# needs no build! per run: refits stay legal (above).

mutable struct AccelCommand
    const structure::AccelStructure
    const mode::Union{Build,Update}
    const deps::Deps                         # the build node's barrier, recorded with it (the one barrier call site)
    const channel::SubmitChannel
    marked::Bool                             # set by a change under the structure's lock; true until first recorded
    cmd::Any                                 # native command buffer, or nothing
    recordedmode::Union{Build,Update}        # what it recorded: an Update before any Build records a Build
    recordedcount::Int                       # what a Build recorded, for builtcount
    lasttoken::Union{Nothing,Token}          # its last submission; the replaced buffer is retired with it
end
AccelCommand(s, mode, deps, ch) = AccelCommand(s, mode, deps, ch, true, nothing, mode, 0, nothing)
mark!(t::AccelStructure) = foreach(c -> (c.marked = true), t.commands)   # under t's lock
# Recorded again with the ordinary builder verbs: no verb of its own. It
# records from the structure's state after the steps of the submission it
# rides in (afterstate: the applied storage, scratch, capacity, total and
# builtcount, with this submission's steps on top, before commit! makes them
# the applied ones), never from the host view, which push! and delete! change
# at their call. Under the structure's lock: every stage locks the SyncStates
# of the structures its AccelCommands build (lockstage; a set fixed per
# plan), so a concurrent push! cannot change `marked` or the slots while it
# records.
function current!(c::AccelCommand, steps)
    c.marked || return c.cmd
    old = c.cmd
    s = c.structure
    st = afterstate(s, steps)
    mode = c.recordedmode = effectivemode(st, c.mode)
    c.recordedcount = commandcount(st, mode)
    b = Builder(s.device, c.channel)         # command objects reserved by prepare! (B.4), so nothing here fails
    # The node's barrier first, on the first prepare kernel (records kernels
    # of a TLAS, the zero-area fill of a BLAS); each later step its own.
    after = emitprepared!(b, preparekernels(s), c.deps)   # c.deps reaches the build if there is no kernel
    emitbuild!(b, nothing, compiledbuild(st, mode, c.recordedcount), after)
    c.cmd = finish(b)
    c.marked = false
    old === nothing || retire!(pool(s), old, (c.lasttoken,))
    return c.cmd
end
commandcount(st, ::Build) = st.total          # the applied total (exact, or the capacity for a kernel-written one)
commandcount(st, ::Update) = st.builtcount
# At submit, under s's lock (internals.jl submitted!): a plan's own Build sets
# builtcount, and a new count marks s's commands like a prefix Build does.
built!(s, ::Build, c) = c.recordedcount == s.builtcount || (s.builtcount = c.recordedcount; mark!(s))
built!(s, ::Update, c) = nothing
# A refit requires the instance count and active set of the last build. Every
# change of the active set (push!, delete!, a source's length) pends a
# BuildAccel(Build()) and marks the structure's commands; the next submission
# that touches the structure applies the Build (coalesced with any pending
# Update into a Build), and the refit's command carries builtcount.

# Changes, through change! (B.4): requests only. One locked call over the
# TLAS (its range table and records share its record) and the range's
# members: the member holds are taken under their locks (holdmember!, B.8b),
# the slot is assigned, the member snapshot and the arrays' reader snapshots
# are replaced, and the range-table entry and a Build are pended. Nothing of
# the TLAS is allocated: a new slot past the range table's size pends a Resize
# of the table (an array like any other, its contents kept), and the capacity
# a new total needs is the applier's (above). The entry's bytes go into an
# upload region of one RangeEntry acquired before the lock (B3.18). A throw
# drops the holds taken so far and the region.
function Base.push!(t::TLAS, blas::BLAS, source::Union{MantleArray,ArrayBox}; scale = 1f0, rotation = nothing)
    checkinstances(t, source)                        # a total past the device's maxInstanceCount throws here
    r = InstanceRange(blas, source, (scale, rotation))
    uses = ((t, Write()), ((m, Read()) for m in holdsof(r))...)   # the range's arrays, a GPURef, the BLAS
    up = acquireupload!(pool(t), sizeof(RangeEntry))
    held = Any[]
    ok = false
    try
        change!(uses) do
            for m in holdsof(r)                          # under each member's lock: the hold and the count together
                holdmember!(m); push!(held, m)
            end
            slot = freeslot!(t)
            t.slots[slot] = r
            foreach(a -> addreader!(a, t), rangearrays(r))   # replaces each array's reader snapshot
            t.members = MemberSet(t)
            marktotal!(t)                                # plans sized from t's total (D.1; B.6)
            mark!(t)                                     # its AccelCommands record a new count
            Change[entrychanges(t, slot, entry(t, slot), up)...;   # Resize of the range table if the slot is new, the entry's Store
                   BuildAccel(nextseq!(t), t, Build())]
        end
        ok = true
    finally
        ok || (foreach(dropcompositehold!, held); retire!(pool(t), up, ()))
    end
    return r
end

function Base.delete!(t::TLAS, r::InstanceRange)
    up = acquireupload!(pool(t), sizeof(RangeEntry))   # the inactive entry's bytes (B3.18)
    ok = false
    try
        change!(((t, Write()), ((a, Read()) for a in rangearrays(r))...)) do
            slot = findslot(t, r)                         # a range not in t (deleted already) throws here
            t.slots[slot] = nothing
            t.members = MemberSet(t)
            marktotal!(t)
            mark!(t)
            # r's arrays keep t as a reader (one count each), and the holds on
            # r's arrays and BLAS stay, until the submission that applies this
            # build has passed: it writes the TLAS, so it waited for every earlier
            # trace (B.4); after it nothing reaches them through t. Until then a
            # write of r's arrays still waits for the plans naming t (internals.jl
            # section 8). Dropped by sweep!, outside every lock.
            defer!(t.sync, tokens -> (foreach(a -> dropafter!(a, tokens, st -> dropreader!(st, t)), rangearrays(r));
                                      foreach(st -> dropafter!(st, tokens, dropcompositehold!), holdsof(r))))
            Change[entrychanges(t, slot, inactiveentry(t, slot), up)...; BuildAccel(nextseq!(t), t, Build())]
        end
        ok = true
    finally
        ok || retire!(pool(t), up, ())
    end
end

# A source's length changes (resize! of a positions array, or of a GPURef's
# target, whose readers include the GPURef and so its TLASes): the Resize is
# pended with the array's readers locked (change!, readers = true), and
# lengthchange pends a Build on each reader TLAS and marks it; the capacity a
# new total needs is allocated by the applier. A store into a source pends an
# Update (`update`, B.5), a refit; `update(b::BLAS, st)` is a Build when st is
# b's index array.

# Capacity: storage and scratch for `capacity` instances, doubling until the
# total fits, starting at 64; it does not shrink on its own, since builds and
# refits carry the exact count and the capacity costs memory only (expected
# from the API: the driver gets the count, not the capacity; not measured with
# storage sized beyond the count, to confirm in phase 5). Growing, hardware: a
# new AS object on a new pool Region (AS storage cannot be sparse) and new
# scratch, prepared by the applier (B.4 step 1); the old ones retired with the
# applying token; the structure's AccelCommands are marked; traces and ray
# queries take the AS by address through the cell (with descriptors instead,
# open decision 4, they would be handle dependents). Software: the five arrays
# grow like any array (their Resize steps, B.5).
# The steps, emitted before the Build's own; commit! sets t.storage, t.scratch
# and t.capacity when the submission is made. The instance records hold
# `capacity` entries and are rewritten by every build, so their move keeps no
# contents.
function capacitysteps(t::TLAS{Hardware}, g::Capacity)
    mark!(t)                                         # AccelCommands record the new storage
    (resizesteps(t.records.state, t.records.state.storage, g.records, Resize(0, t.records.state, (g.capacity,), 0))...,
     CellWrite(cellstate(t), acceladdress(t.device, g.as), ()),
     MoveAccel(t, t.storage, g.as, t.scratch, g.scratch))
end
capacitysteps(t::TLAS{Software}, g::Capacity) = softwaresteps(t, g)   # the five arrays' Resize steps (D.1)
# storage, scratch and capacity are as applied (like an array's storage,
# B.2): only an applying submission changes them, at commit. Applying
# MoveAccel retires the old storage and scratch with its token.

# free!(t) drops the creator's hold; at zero, its ranges' composite holds on
# arrays and BLASes are dropped with the TLAS's last tokens and its storage
# retired. Its finalizer does the same through the pool's inbox.

# A BLAS: the same shape as a TLAS one level down. Storage on a pool Region
# sized from its geometry, with a capacity that doubles; it holds its vertex
# and index arrays (composite holds) and is their reader. A new topology (the
# geometry's length changed) is a Build into its storage, a MoveAccel to new
# storage beyond its capacity (a new address through its cell); moved
# vertices with the same topology are an Update, new indices a Build (an
# update may not change them, header). Either way the TLASes over
# it (its readers) get an Update: the records kernel reads the BLAS address
# from its cell, and an update may change which BLAS an instance references
# (header). Shared between TLASes through its geometry: one BLAS per marker
# mesh.

# ── B.8b GPURef of an array ───────────────────────────────────────────────────

# GPURef(dev, a::MantleArray) (Simon, 2026-10-02): the GPURef for an array.
# Like the GPURef for a value, a host-side box whose content the next run
# sees; here the content is an array, replaceable between runs without
# recompiling. RayMakie's plot attributes are GPURefs of arrays, so a plot made
# from a device array reads that array (zero-copy) and
# update!(plot; positions = other) replaces it (recording-plan.md, Graphics).
# A composite with one member: it holds its target (a composite hold), is
# one of its target's readers, and has readers of its own (TLASes with a
# range over it).
abstract type GPURef end
# The value form, ValueBox{T} <: GPURef, is internals.jl section 7: copied into
# the run's entry, a scalar a kernel takes by value.
mutable struct ArrayBox{T,N} <: GPURef
    const device::Device                # device-only: its target is a device array
    target::MantleArray{T,N}            # host view (r[]), replaced under the lock at the call
    applied::MantleArray{T,N}           # what its cell points at: set when the Rebind's step commits, under
                                        #   the sync locks of the plans naming the box (the applier holds them);
                                        #   a run resolves a write through the box to it (internals.jl section 8)
    members::MemberSet                  # snapshot (target,), as a TLAS's (B.4 lockexpanded)
    const cell::Cell                    # holds the target's cell index: the head reads through it
    const sync::SyncState
    const holds::Threads.Atomic{Int}
    readers::ReaderSet                  # TLASes with a range over it; an immutable snapshot (B.2)
    const dependents::Dict{WeakPlan,Dependence}   # plans that bound its target by handle (B.6)
end

# A member's hold is taken under the member's record lock, inside the locked
# call of the change that adds it (push!, a rebind, GPURef itself): the hold
# and the composites count change together, so evictable, checked under the
# same lock, never sees one without the other. A member evicted before it
# became one is restored by the next submission that uses the composite (B.4
# step 1 covers the composites' members), not here: the call allocates
# nothing. Throws for a freed or released member (nothing changed yet: apply
# calls it first).
holdmember!(st::ArrayState) = (usable(st); addcompositehold!(st))   # internals.jl section 10
holdmember!(x) = addhold!(x)        # a BLAS or an array GPURef: never evicted

# newbox creates the box (its cell and record): creating an object, like
# creating an array; the cell write of its target is pended like a rebind.
function GPURef(dev::Device, a::MantleArray{T,N,<:Device}) where {T,N}
    rootonly(a)
    r = newbox(dev, a)                  # its cell and record; no hold, no reader entry, no finalizer yet
    held = false
    ok = false
    try
        change!(((r, Write()), (a.state, Read()))) do
            holdmember!(a.state)
            held = true
            addreader!(a, r)            # the box is one of its target's readers
            (Rebind(nextseq!(r), r, a),)
        end
        finalizer(x -> push!(pool(x).inbox, DropHolds(x)), r)   # only now: it releases what the box holds
        ok = true
        return r
    finally
        ok || (held ? release!(r) : discard!(r))   # a throw: what the box took goes back, once (no finalizer yet)
    end
end
discard!(r::ArrayBox) = retire!(pool(r), r.cell, ())   # a box that never held anything: its cell (and record)
GPURef(dev::Device, v) = ValueBox(dev, v)
# The box's cell holds a cell index, not a view's offset and strides, so its
# target is a root array: a view throws.
rootonly(a::MantleArray) = rootonly(a.role)
rootonly(::Root) = nothing
rootonly(::Derived) = throw(ArgumentError("a GPURef of an array targets a root array, not a view"))

# r[] = b: same element type and dimension count (the shader or kernel type is
# unchanged, so nothing recompiles); b is a root device array. A request like
# a store: r[] reports b at once; the submission that applies it writes r's
# cell; runs submitted before still read the old target, whose hold drops once
# the applying submission has passed. The box moves from old's readers to
# b's; its own readers (TLASes) get an Update, or a Build if the length
# differs (capacity, if needed, is the applier's). The parts of plans that
# bound r's target by handle (an index buffer through the GPURef,
# r.dependents) are marked for recording again when the Rebind is applied
# (open decision 17 (c); B.4). Rebinding to the current target does nothing
# (checked without the lock: a racing rebind decides either way).
function Base.setindex!(r::ArrayBox{T,N}, b::MantleArray{T,N,<:Device}) where {T,N}
    r.target === b && return r
    rootonly(b)
    held = false
    ok = false
    try
        change!(((r, Write()), (b.state, Read()))) do    # r brings its target
            usable(r)                                    # a freed GPURef throws
            holdmember!(b.state)
            held = true
            old = r.target
            cs = Change[Rebind(nextseq!(r), r, b);
                        (rebound(t, length(old), length(b)) for t in r.readers)...]   # Update, or Build + mark!
            r.target = b                                 # what r[] reports
            r.members = MemberSet((b,))                  # a new snapshot
            addreader!(b, r)                             # the box follows its target
            # The old target keeps the box as a reader, and its hold, until
            # the submission applying the Rebind has passed (it writes the
            # box, so it waited for every earlier reader; as delete!, B.8).
            defer!(r.sync, tokens -> (dropafter!(old.state, tokens, st -> dropreader!(st, r));
                                      dropafter!(old.state, tokens, dropcompositehold!)))
            cs
        end
        ok = true
    finally
        ok || (held && dropcompositehold!(b.state))      # taken, then the change threw before it was pended
    end
    return r
end
stepsof(c::Rebind, p) = (CellWrite(cellstate(c.target), cellindex(c.new), ()), RecordAgain(c.target.dependents))
committed!(c::Rebind) = (c.target.applied = c.new)   # by commit!, under the applier's locks
# The box's last hold (its creator's free!, its finalizer, the last range or
# plan holding it): under its lock, once the box's last accesses have passed,
# it leaves its target's readers and the target's composite hold drops; its
# cell and record go back with the same tokens.
release!(r::ArrayBox) = locked((r.sync, r.target.state.sync); lent = true) do
    tokens = lasttokens(r.sync, r)
    dropall!(r.sync.pending); empty!(r.sync.pending)   # a CellWrite no submission took
    foreach(d -> d(tokens), r.sync.deferred); empty!(r.sync.deferred)   # an earlier target's hold
    dropafter!(r.target.state, tokens, st -> dropreader!(st, r))   # as a rebind
    dropafter!(r.target.state, tokens, dropcompositehold!)
    retire!(pool(r), r.cell, tokens)
end
Base.getindex(r::ArrayBox) = r.target

# Launches and draws over an ArrayBox are always indirect (LengthOf(r) never
# resolves to Direct: the target can change); its slots get address and dims
# from the target's cell, read through r's cell by the head. Device side it is
# the same type as the target's devicearg, so one pipeline serves both.

# ── B.9 Residency and textures ────────────────────────────────────────────────

# Metal: every pool block, reserve and acceleration structure is in one
# residency set per device, added to every Mantle queue (addResidencySet);
# rawalloc/rawfree and binds add and remove allocations and commit. A BLAS
# reached only through a TLAS is resident without any plan naming it. Per
# plan: its ICBs, its argument memory and the pipelines its ICBs name.

# Textures: a device-wide table (Vulkan: a descriptor set with
# update-after-bind, partially bound, update-unused-while-pending bindings, or
# a descriptor buffer; Metal: an argument buffer of texture resource IDs). A
# texture's index is a value in its cell; shaders read it from their slots.
# A texture written in place keeps its index. A regrown or replaced texture
# gets a new, unused index, scheduled like any change: the applying
# submission's prepare! creates the new image (texels kept where the sizes
# overlap, decision pending), its descriptor is written into the unused index
# before the prefix (no pending command buffer uses that index,
# descriptorsets.adoc:793-807), the cell write is a step, and the old index
# and image are retired with that submission's token, then reused. A restored image (internals.jl
# section 9) gets a new index the same way. The table and its free list are
# the device's, under the table's own lock (a leaf).

# ── B.10 Free and leaks ───────────────────────────────────────────────────────

# free!(a) through the creator's handle: marks the state freed under its lock
# (later calls through any handle throw: eager calls, declarations, stores)
# and drops the creator's hold. Runs of graphs that still hold the array do
# not check `freed`. At the last hold (internals.jl section 10, release!):
# under the record's lock the pending changes are discarded (sources retired
# with no tokens, prepared storage released), deferred hold drops are called
# with the record's last tokens, the storage becomes Released(), the old
# storage and the cell are retired with those tokens, and the state leaves
# the eviction list. Every holder type (ArrayBox, TLAS, BLAS, a RenderObject)
# has a finalizer that pushes its hold drops through the pool's inbox.

# ══ C. What a device type implements ══════════════════════════════════════════

caps(dev)                    # facts, no policy: .sparseresidency, .sparsepage, .sparseaddressspace,
                             #   .accelbyaddress, .separatehostheap (eviction frees device memory),
                             #   .display (drives a monitor), .inlinestore and .inlinealign (B.5);
                             #   core's sparsethreshold, reservesize (B.1) and submission budget are
                             #   computed from them
countsource(dev)             # CountRecorded() (Vulkan) | CountFromMemory() (Metal): where a TLAS build gets its count
createreserve(dev, bytes)    # -> Reserve
bindpages!(ch, token, binds, waits)     # queue operation: vkQueueBindSparse | Metal sparse buffer mapping update (MTL4UpdateSparseBufferMappingOperation in Metal.jl; selector to check on the SDK) | cuMemMap; signals token
unbindpages!(ch, token, unbinds, waits) # the same, unbinding
instancerecordtype(dev)      # VkAccelerationStructureInstanceKHR | MTLIndirectAccelerationStructureInstanceDescriptor
accelsizes(dev, path, n)     # storage and scratch for n instances or a geometry
placeaccel(dev, path, region)        # AS object on a pool Region
acceladdress(dev, as)        # vkGetAccelerationStructureDeviceAddressKHR | gpuResourceID
emitbuild!(b, node, cmd::CompiledBuild, deps)   # Build or Update; the exact count (Vulkan, in an AccelCommand) or from a cell (Metal)
emitstore!(b, dst, bytes)                       # a small store in a change prefix: vkCmdUpdateBuffer; Metal, CUDA,
                                                #   HIP: an upload region prepare! acquired before the locks (B.3, B.4)
emitdraw!(b, node, cmd::CompiledDraw)           # inside beginrender!/endrender!; direct or indirect, as compile decided
bindindices!(b, node, storage)                  # vkCmdBindIndexBuffer on the storage's buffer (open decision 17)
tableentry!(dev, table, index, texture)         # a host write to an unused index (not a recorded command)

# Software acceleration structures implement the same core calls with kernels
# (D.1); `path = Software()` dispatches there.

# ══ D. Consumers ══════════════════════════════════════════════════════════════

# ── D.1 Raycore: the software TLAS ────────────────────────────────────────────

# Storage: five device arrays, resized in place, never replaced.
struct SoftwareStorage
    nodes::MantleArray{BVHNode2,1}
    instances::MantleArray{InstanceDescriptor,1}
    blasnodes::MantleArray{BVHNode2,1}
    blasprims::MantleArray{Triangle,1}
    blasdescs::MantleArray{BLASDescriptor,1}
end

# The build is kernels over the instance count, with temporaries sized from
# the capacity; bounds reduction on the device, no readback. sortperm! is
# Mantle's (AcceleratedKernels' kernels launched through dispatch!), so it
# works on graph arrays.
function buildaccel!(l, t::TLAS{Software}, ::Build)
    s = t.storage; n = LengthOf(t.records)
    MantleArray(l, UInt32, SizedBy(identity, t.records)) do codes
    MantleArray(l, UInt32, SizedBy(identity, t.records)) do perm
        bounds = MantleArray(l, AABB, 1)
        dispatch!(l, instancebounds!, (bounds, t.records), n)
        dispatch!(l, mortoncodes!, (codes, t.records, bounds), n)
        sortperm!(perm, codes)
        dispatch!(l, emittopology!, (s.nodes, codes, perm), n)
        dispatch!(l, refitnodes!, (s.nodes, t.records), n)
    end end
end
buildaccel!(l, t::TLAS{Software}, ::Update) = dispatch!(l, refitnodes!, (t.storage.nodes, t.records), LengthOf(t.records))

# Kernels get the TLAS as a struct of five device arrays; each has its own
# entry in the head's cell table, at any nesting depth. Traversal of an empty
# TLAS returns a miss (length(nodes) == 0 from the slot); today it reads
# nodes[1] out of bounds (Raycore instanced-bvh.jl:1951-1956).
devicearg(dev, t::TLAS{Software}) = StaticTLAS(map(a -> devicearg(dev, a), fields(t.storage))...)

# Gone: Raycore's TLAS finalizer that finalize()s arrays a recording or
# StaticTLAS still uses (instanced-bvh.jl:372, 400-411; holds replace it);
# blas_array (device pointers stored in device memory, read only for
# root_aabb); host readbacks and synchronize() between build steps.

# ── D.2 Hikari ────────────────────────────────────────────────────────────────

# The scene owns one TLAS. Scene changes are pending TLAS changes (from host
# calls, or pended by runs of other graphs that write the TLAS's source arrays,
# B.8; open decision 16, ruled B3.9/B3.10). The integrator is declared once into the
# screen's frame graph (one graph per screen), outside any plot's insert!
# block, and kept across scene and plot changes (a plot is a piece of the
# raster pass and recompiles nothing, B3.17). Its film and sample count are device
# arrays the screen holds, so accumulation survives runs and recompiles; its
# work queues are window-sized graph arrays, so a window resize within the
# capacity records the window's segments again (internals.jl section 7) and
# declares nothing. It holds no build or refit node: the pending update is
# applied by the first submission that touches the TLAS, so a frame with no
# change does no acceleration-structure work. Only a scene whose transforms a
# kernel of this graph writes declares refit!(g, scene.tlas); a kernel-written
# count needs no build! per run (B.8: built over the capacity, refits stay
# legal).
function declare!(g, vp::VolPath, scene)
    fill!(vp.active, Int32(1))
    repeat!(g, vp.maxdepth ÷ 2; while_nonzero = vp.active) do i
        trace!(g, vp.pipeline, scene.tlas, (vp.queue, vp.hits), launchrange(vp.queue))
        dispatch!(g, shade!, (vp.hits, vp.queue, lengthof(vp.queue), vp.active, scene.materials, scene.tlas),
                  launchrange(vp.queue))         # writes the queue's next length and active = length > 0
    end
end
# Gone: notify_scene_changed dropping the integrator's plans (Hikari
# scene.jl:141-151, from nine scene mutators and on accel.dirty);
# fill_aux_buffers!'s waitidle (hw-rt.jl:104). Raycore.sync! inside
# Adapt.adapt_structure (Raycore instanced-bvh.jl:1117, Mantle
# vulkan/raytracing/hwtlas.jl:243-244, metal/hwtlas.jl:325) goes too.

# ── D.3 RayMakie ──────────────────────────────────────────────────────────────

# A plot attribute is a GPURef of a MantleArray (B.8b): length 1 for one
# value, n for one per element. Shaders read attribute(a, i) =
# a[min(i, length(a))] (name open), so both are one pipeline. A change that
# alters a shader (an element type, a material type, transparency) is open
# decision 18. A RenderObject holds its GPURefs and arrays; its finalizer drops
# them through the pool's inbox.
struct RenderObject
    pipeline::GraphicsPipeline                 # with the arguments' types: the pipeline key
    attributes::NamedTuple                     # name => GPURef of an array; created once with the plot
    owned::NamedTuple                          # name => the plot's own MantleArray (host data lands here)
    visible::MantleArray{UInt32,1}             # the when! flag around its draw (a store: no recompile)
    rect::MantleArray{Vec4f,1}                 # the scene's rectangle, shared by the scene's plots (one store
end                                            #   per layout change), applied in the vertex stage

# update!(plot; …) calls this once per changed attribute (ComputePipeline's
# changed set); everything is pending and lands in the next frame's submission.
setattribute!(r::RenderObject, name, data::AbstractArray) =                # host data
    (own = r.owned[name]; copy!(own, data); r.attributes[name][] === own || (r.attributes[name][] = own))
setattribute!(r::RenderObject, name, data::MantleArray) =                  # a device array: zero copy
    (r.attributes[name][] = data)
setattribute!(r::RenderObject, name, value) =                              # one value
    setattribute!(r, name, [value])                                        # a one-element store (inline, B.3)
setvisible!(r::RenderObject, v::Bool) = (r.visible[1:1] = [UInt32(v)])   # the head multiplies the draw's count by it

# The frame graph, declared once per screen. Each plot is a piece of the
# pass (decisions B3.17): inserting or deleting one compiles and records that
# piece only and rewrites the pass's list; the plan is not recompiled, and
# every other plot's recording stays. Each draw keeps its own pipeline and
# index buffer, so a large mesh draws as it would alone.
function declareframe!(g, screen)
    screen.pass = render!(g, screen.window => Clear(screen.background), screen.depth => Clear(1f0)) do p
    end                                                                   # plots come as pieces
end
insertplot!(screen, r::RenderObject) =                                    # Makie's insert!
    r.piece = insert!(screen.pass) do                                     # appended: the list order is the draw order
        when!(screen.pass, r.visible) do
            draw!(screen.pass, r.pipeline, (values(r.attributes)..., r.rect, screen.camera),
                  launchrange(r.attributes.positions))                   # indirect: a GPURef of an array
        end
    end
deleteplot!(screen, r::RenderObject) = delete!(screen.pass, r.piece)     # Makie's delete!

# The render loop: no walk over plots or render objects.
function renderloop!(screen)
    while isopen(screen.window)
        foreach(resolve!, takedirty!(screen))       # only plots whose inputs changed (their edges put them there)
        run!(screen.framegraph)                     # pending changes, head, recorded draws, present
        # A plot added or deleted (insertplot!, deleteplot!) is applied by
        # this run before its stages: one piece compiled and recorded, the
        # pass's list rewritten (internals.jl section 5, B3.17).
    end
end

# A plot from a simulation's device arrays: mesh!(vertices, faces) makes the
# array GPURefs point at them. Each run of the simulation writes them; the frame
# draws over their current lengths. The frame's run reads them through the
# GPURefs, so it waits for the simulation's write, and the simulation's next
# write waits for the frame (B.4, lockexpanded). The screen is one of their
# readers, so a run that writes them marks it for redraw (internals.jl
# section 6; open decision 16, ruled B3.9/B3.10). Faces are an index buffer bound by
# handle: open decision 17.

# Glyph atlas: a Texture2D in the device table; a larger atlas gets a new
# index through its cell. Glyph UVs are stored in texels and normalized in the
# shader by the atlas's size read at run time, so atlas growth touches no
# text or scatter plot (today the UVs are normalized and the atlas width is a
# compile-time constant, RayMakie overlay/scatter.jl:195, :426).
# Today (RayMakie e6e1aa9b0, overlay_rendering.jl:234-256) a frame plan is
# rebuilt when the set of plots, visibility, a pipeline or an argument type
# changes, and frames are walked, not recorded (:398-402). Gone: the walked
# frames, DrawBinding/rebind!, repacking argument memory on the host every
# frame, frame_signature, the per-frame scene-tree walks.

# meshscatter (ray traced): one BLAS per marker mesh (shared through the
# geometry arrays, held by every range that uses it), one instance range
# following the positions.
function meshscatter_instances!(tlas, plot)
    marker = blas!(plot.scene, plot.marker[])        # the scene's BLAS for this geometry (name open)
    push!(tlas, marker, plot.attributes.positions; scale = plot.attributes.markersize,
          rotation = plot.attributes.rotation)
end
# positions is the plot's array GPURef: the TLAS is its reader, and the
# GPURef a reader of its target, so a change of the target reaches the TLAS
# (B.8b). Moving particles, same count: a store into the target pends a TLAS
# refit. A new count: the resize! pends a build with the exact count (storage
# beyond the capacity is allocated by the submission that applies it, B.4);
# the BLAS stays.
# Another device array (update!(plot; positions = b)): a rebind, the TLAS
# gets an Update, or a Build if the length differs. Hidden: mask 0 in the
# range's entry, a refit. A colour: a store into the material table.

# Per frame, steady state: pending changes applied at the start of the frame's
# submission, the head (1.3-3.5 µs measured, v2_*.txt), indirect draws, a
# refit or build only if one is pending, the trace. No recompile and no
# re-record. The host waits only where a value GPURef in the entry changed
# (the previous run must have finished, plan Run); per-frame values such as
# the camera are one-element arrays written by stores, which never wait.
