# Compilation is a list of phases, not a function.
#
# RPS splits the same work six ways (`src/runtime/common/phases/`), and the
# reason is not tidiness: each analysis becomes testable on its own. Ours grew
# four interacting analyses inside one function, and fuzzing found three bugs
# that all lived in the interaction rather than in any one of them.
#
# A phase reads and writes named fields on a shared context. The backend adds the
# concrete phases; the core defines the protocol and the order.

abstract type Phase end

"""
    run!(phase, ctx)

Advance the compilation by one phase. Every phase is expected to be callable on
its own against a context carrying only what earlier phases produced, which is
what makes each one testable without running the others.
"""
function run!(phase::Phase, ctx)
    throw(MethodError(run!, (phase, ctx)))
end

# ── what a backend's compilation context has to answer ────────────────────────
#
# Five of the seven phases are graph analyses and contain nothing backend-shaped;
# only Barriers and Pipelines do. They used to live in the Lava extension anyway,
# which meant a second backend had to copy a scheduler — and two copies of a
# scheduler drift in exactly the places fuzzing found bugs. They live here now,
# and a backend supplies the five methods below.

"""
Everything the backend-independent phases compute.

One struct rather than a field-per-phase interface: a context holds one of these
and answers [`analysis`](@ref) with it, so adding a phase output does not add an
accessor pair to every backend.
"""
mutable struct Analysis
    deps::Vector{Vector{Int}}                   # Dag
    order::Vector{Int}                          # Schedule
    items::Vector{Item}                         # Liveness
    peak::Int                                   # Place
    naive::Int
    offsets::Vector{Int}
    # Aliasing: pass index => the (new, old) transient pairs whose bytes change
    # hands there. The old index is kept because the barrier is derived from what
    # that transient was last doing, not assumed to be everything.
    alias_begins::Dict{Int,Vector{Tuple{Int,Int}}}
    # Place: the SHARED region of each arena this plan was placed into, and which
    # arena that was. Not owned — every tenant of an arena holds the same region —
    # so teardown deregisters rather than releasing, and the last tenant out is
    # what actually gives the bytes back.
    regions::Vector{Any}
    arenas::Vector{Any}
end
Analysis() = Analysis(Vector{Int}[], Int[], Item[], 0, 0, Int[],
                      Dict{Int,Vector{Tuple{Int,Int}}}(), Any[], Any[])

"""
    Compilation

The state a compilation carries between phases, as far as the phases are
concerned. A backend subtypes this for its own context.

**Six of the accessors below then come for free**, because they were the same
line in every backend. Both extensions had written, verbatim:

    analysis(c)      = c.analysis
    passes(c)        = c.graph.passes
    policy(c)        = c.policy
    alias(c)         = c.alias
    transients(c)    = c.graph.transients
    transientbyid(c) = c.graph.transient_by_id
    device(c)        = c.graph.dev

Field access rather than a method per backend, and that is a real contract rather
than a convenience: a `Compilation` **must** have `analysis`, `alias` and
`policy` fields and a `graph` field whose value has `passes`, `transients`,
`transient_by_id` and `dev`. A backend whose context is shaped differently
overrides the accessor, which is ordinary dispatch and costs it nothing.

What stays per backend is what actually differs: `pool` (where the memory comes
from), `overlapping` (whether this backend has slices), and `usages` (how a pass
records them).
"""
abstract type Compilation end

"""The [`Analysis`](@ref) a compilation context carries."""
function analysis end
analysis(c::Compilation) = c.analysis

"""The passes a context is compiling, in DECLARATION order."""
function passes end
passes(c::Compilation) = c.graph.passes

"""A pass's declared resource usages, as `id => Usage` pairs."""
function usages end

"""
    overlapping(ctx, a::Int, b::Int) -> Bool

Whether two resource ids can name the same bytes.

Equal ids do, and so does any slicing relationship a backend supports — a slice
against its own parent, two intersecting slices of one parent. A scheduler told
that two such ids are unrelated is free to reorder two passes that write the same
memory, so a backend with no slices still has to answer `a == b`.
"""
function overlapping end

"""The scheduling [`Policy`](@ref) a context was built with."""
function policy end
policy(c::Compilation) = c.policy

"""The device a context compiles for. Where its [`Pool`](@ref) lives."""
function device end
device(c::Compilation) = c.graph.dev

"""
The device's [`Pool`](@ref) — one per memory kind, owned by the device and never
by the caller. `Place` suballocates from it instead of allocating per compile.
"""
function pool end

"""
Smallest block the pool takes from this device.

A device property rather than a Mantle constant: 64 MiB is right for discrete
VRAM, where an allocation is expensive and a driver has a limit on how many you
may hold, and wrong for host memory, where over-allocating just faults in pages
nobody asked for.
"""
blocksize(dev) = 64 << 20

"""Whether transients may share bytes. `false` gives every one the whole
timeline, which is a bisection tool rather than a tuning knob — it is how the
aliasing hazard was isolated."""
function alias end
alias(c::Compilation) = c.alias

"""Every transient the graph declared, in declaration order."""
function transients end
transients(c::Compilation) = c.graph.transients

"""`id => transient` for the ids passes name them by."""
function transientbyid end
transientbyid(c::Compilation) = c.graph.transient_by_id

# Transient accessors. A backend answers these for whatever it uses to represent
# one; the phases never look inside.
"""Bytes a transient occupies."""
function nbytes end
"""Alignment a transient's offset must satisfy."""
function alignment end
"""Which allocation a transient belongs in. Transients in different arenas never
share bytes, so offsets are only comparable within one."""
function arena end
"""A transient, for an error message."""
function describe end

"""
    materialize!(device, transient, slab, offset)

Give a placed transient its storage, at `offset` in `slab`. The device is the
graph's — a transient's driver storage belongs to the device that made it, and
on more than one device the process default is not necessarily that device.
"""
function materialize! end

"""Passes in execution order — the scheduled order once Schedule has run,
declaration order before that."""
ordered(c) = (o = analysis(c).order; isempty(o) ? passes(c) : passes(c)[o])

"""
Which pass depends on which, from the resources they share.

Pass j depends on pass i when they touch a common resource and at least one of
them writes it. Read-after-read is not an edge, which is what leaves independent
work free to be reordered.
"""
struct Dag <: Phase end

function run!(::Dag, c)
    ps = passes(c)
    a = analysis(c)
    a.deps = [Int[] for _ in ps]
    for j in eachindex(ps), i in 1:(j - 1)
        shared = false
        for (idj, Uj) in usages(ps[j]), (idi, Ui) in usages(ps[i])
            overlapping(c, idi, idj) || continue
            (writes(Ui) || writes(Uj)) || continue
            shared = true
            break
        end
        shared && push!(a.deps[j], i)
    end
    return c
end

"""
Choose an execution order consistent with the DAG.

RPS spends its largest phase here (1171 lines) and encodes the whole policy as
bit *positions* in one 32-bit score, so competing objectives compare with a
single integer and retuning means moving a bit rather than rewriting a
comparator. The policy types below pick those positions.
"""
struct Schedule <: Phase end

abstract type Policy end

"""Prefer passes that free more bytes than they allocate: smaller peak, more
ordering between passes, which is RPS's `RPS_SCHEDULE_PREFER_MEMORY_SAVING_BIT`."""
struct Compact <: Policy end

"""Prefer declaration order and leave lifetimes long: larger peak, fewer
dependencies forced by shared memory. RPS's default."""
struct Overlap <: Policy end

"""Which bit range each objective occupies. Moving these is how the policy is
tuned, exactly as in `rps_dag_schedule.hpp:180-208`."""
memory_shift(::Compact) = 16
memory_shift(::Overlap) = 0
order_shift(::Compact) = 0
order_shift(::Overlap) = 16

"""Width of each memory term in the packed score; two of them share the range."""
const SCORE_MAX = 0x1fff

"""
Greedy list scheduling under a packed priority score.

Objectives live in disjoint bit ranges of one integer and are OR'd, so comparing
candidates is one integer compare and the policy is chosen by moving a bit range
rather than by writing a different comparator. That trick is the best single idea
in RPS (`rps_dag_schedule.hpp:180-208`).

The memory term prefers a candidate that frees more bytes than it allocates,
counting a transient's first touch as an allocation and its last as a free.
"""
function run!(::Schedule, c)
    ps = passes(c)
    n = length(ps)
    n == 0 && return c

    deps = analysis(c).deps
    remaining = [length(d) for d in deps]
    dependents = [Int[] for _ in 1:n]
    for j in 1:n, i in deps[j]
        push!(dependents[i], j)
    end

    # How many *passes* touch each transient, so "last use" is known. Counting
    # usages instead would double-count a pass that binds the same resource
    # twice, which makes every candidate tie and the schedule collapse to
    # declaration order.
    ids(p) = unique(first(u) for u in usages(p))
    touches = Dict{Int,Int}()
    for p in ps, id in ids(p)
        touches[id] = get(touches, id, 0) + 1
    end
    seen = Dict{Int,Int}()
    bytesof = Dict{Int,Int}()
    for (id, t) in transientbyid(c)
        bytesof[id] = nbytes(t)
    end

    maxalloc = max(1, maximum(eachindex(ps); init = 0) do i
        sum(id -> get(bytesof, id, 0), ids(ps[i]); init = 0)
    end)
    mshift = memory_shift(policy(c))
    oshift = order_shift(policy(c))
    order = Int[]
    ready = [i for i in 1:n if remaining[i] == 0]

    while !isempty(ready)
        best, bestscore = 0, -1
        for k in eachindex(ready)
            i = ready[k]
            alloc = 0
            freed = 0
            for id in ids(ps[i])
                b = get(bytesof, id, 0)
                b == 0 && continue
                get(seen, id, 0) == 0 && (alloc += b)
                get(seen, id, 0) + 1 == touches[id] && (freed += b)
            end
            # Two non-negative terms, not their difference. A difference has to
            # be clamped at zero to fit the bit range, and that clamp throws away
            # exactly the distinction being measured: a pass that breaks even and
            # one that allocates two buffers both come out at zero.
            #
            # Scaled by the largest per-pass allocation rather than by a fixed
            # shift. RPS shifts by 16 (`rps_dag_schedule.hpp:351`) because its
            # buffers are megabytes; ours can be any size, and a fixed shift sends
            # every 16 kB buffer to zero and collapses the schedule back to
            # declaration order.
            mem = clamp((SCORE_MAX * (maxalloc - alloc)) ÷ maxalloc, 0, SCORE_MAX) +
                  clamp((SCORE_MAX * freed) ÷ maxalloc, 0, SCORE_MAX)
            # Clamped because the fields are OR'd, not added: a term that spills
            # its 16 bits would silently corrupt the one above it.
            ord = clamp(n - i, 0, 0xffff)                # earlier declaration scores higher
            score = (mem << mshift) | (ord << oshift)
            if score > bestscore
                best, bestscore = k, score
            end
        end
        i = ready[best]
        deleteat!(ready, best)
        push!(order, i)
        for id in ids(ps[i])
            seen[id] = get(seen, id, 0) + 1
        end
        for j in dependents[i]
            remaining[j] -= 1
            remaining[j] == 0 && push!(ready, j)
        end
    end

    length(order) == n ||
        error("scheduling left $(n - length(order)) passes unreachable: cycle in the DAG")
    analysis(c).order = order
    return c
end

"""Which passes a resource is live across. Inputs to placement."""
struct Liveness <: Phase end

"""
Intervals come from which passes touched a resource, recorded as the graph was
built. Liveness is therefore never a second statement that could disagree with
use.

With `alias = false` every transient claims the whole timeline, so none can share
bytes. That is not a tuning knob so much as a bisection tool: it is how the
aliasing hazard was isolated.
"""
function run!(::Liveness, c)
    ts = transients(c)
    isempty(ts) && return c
    byid = transientbyid(c)
    # Recomputed from the scheduled order, not from declaration order: reordering
    # passes is exactly what changes a transient's interval.
    for t in ts
        t.first, t.last = typemax(Int), 0
    end
    for (pos, p) in enumerate(ordered(c)), (id, _) in usages(p)
        t = get(byid, id, nothing)
        t === nothing && continue
        t.first = min(t.first, pos)
        t.last = max(t.last, pos)
    end
    # A transient nothing touched has no interval, and the placer's `Span` reports
    # that as an inverted range with `typemax(Int)` in it — a message about the
    # allocator for a mistake in the graph.
    for (i, t) in enumerate(ts)
        t.last == 0 && throw(ArgumentError(
            "transient $i ($(describe(t))) is never used by any pass. A transient's " *
            "interval is derived from use, so one that nothing reads or writes has " *
            "nothing to place. Give it to a pass, or drop the declaration."))
    end
    lastpass = maximum(t -> t.last, ts) + 1
    analysis(c).items =
        [Item(string(i), alias(c) ? Span(t.first, t.last + 1) : Span(0, lastpass),
              nbytes(t); alignment = alignment(c, t))
         for (i, t) in enumerate(ts)]
    return c
end

"""Offsets for every transient, from the liveness intervals.

Named `Place` because `Placement` is already the *result* of `place`."""
struct Place <: Phase end

"""
Assign offsets, one placement per arena.

Peak is summed over arenas because they are separate allocations; the alternative
would be to report the larger and pretend the other is free.
"""
function run!(::Place, c)
    a = analysis(c)
    isempty(a.items) && return c
    ts = transients(c)
    a.offsets = zeros(Int, length(ts))
    a.peak = 0
    a.naive = sum(nbytes, ts)
    for ar in unique(map(arena, ts))
        idx = findall(t -> arena(t) == ar, ts)
        # Bounded by what the device can actually give (see `headroom`), and
        # checked here rather than at `rawalloc`: this is the only point that
        # still knows the ITEMS, and "over budget by 40 MB" is answerable by
        # dropping one buffer and unanswerable without knowing which. Finding out
        # at allocation time instead attributes the failure to whatever allocated
        # next.
        prob = Problem(a.items[idx], headroom(pool(c), device(c), ar))
        pl = checkcapacity(prob, place(prob), "arena $ar")
        # `reserve!`, not `acquire!`: every plan placed in an arena gets the SAME
        # region, so the arena costs the largest of them instead of their total.
        # A private slice per plan cannot share bytes however well either one is
        # placed, which is the property a device-owned arena exists to have.
        # The REGION has to be at least as aligned as anything placed in it. The
        # placer aligns each tenant WITHIN the region, so a region that starts
        # at a weaker boundary shifts every one of those offsets off theirs —
        # and the sum is what the backend is handed.
        #
        # It defaulted to `reserve!`'s 256, which held only because nothing had
        # ever asked for more. A Metal render target wants 2048, and the second
        # one in an arena landed 128 bytes short of its boundary: the driver
        # refuses to place a texture there, which is at least loud. The Vulkan
        # side would have bound the image to a misaligned offset.
        want_align = maximum(it -> it.alignment, a.items[idx]; init = REGION_ALIGN)
        want_align = max(want_align, REGION_ALIGN)
        reg = reserve!(pool(c), device(c), ar, ts[idx], pl.height;
                       align = want_align, blocksize = blocksize(device(c)))
        push!(a.regions, reg)
        push!(a.arenas, ar)
        a.peak += pl.height
        for i in idx
            a.offsets[i] = pl.offsets[string(i)]
            # The transient's offset is relative to the REGION; the region's is
            # relative to the block. Adding them is where a suballocated plan
            # differs from one that owns its allocation outright.
            materialize!(device(c), ts[i], memoryof(reg), offset(reg) + a.offsets[i])
        end
    end
    return c
end

"""
    giveup!(pool, regions, arenas, x)

Give up `x`'s claim on the arenas it was placed into.

Explicit, and never a finalizer — see [`trim!`](@ref). Deregistering rather than
releasing, because the region is SHARED: the bytes go back when the last tenant
does, and a plan that released them itself would pull them out from under
whichever other plan is still placed there. A plan dropped without this holds an
arena at its size until the pool is trimmed — a leak rather than a crash, and the
right way round.

Takes the two vectors rather than an `Analysis`, because a compiled plan keeps
them and has thrown the analysis away. Both backends' `free!` were this, spelled
out.
"""
function giveup!(pool, regions, arenas, x)
    for ar in arenas
        untenant!(pool, ar, x)
    end
    empty!(regions)
    empty!(arenas)
    return nothing
end

"""
Which passes begin using memory another transient just finished with.

Separate from placement because it is a *consequence* of placement that the
barrier phase needs, and because it is the one hazard per-resource tracking
cannot see: two aliased transients look unrelated to it.
"""
struct Aliasing <: Phase end

"""
Which passes begin a transient that took over another's bytes.

Aliasing creates a hazard between two resources the usage tracker sees as
unrelated: Y's first write lands on memory X was still reading. Nothing in the
per-resource state can know that, so placement has to say so. RPS carries the
same thing as ResourceAliasingInfo with srcDeactivating / dstActivating.

Found by fuzzing: with aliasing on, derived barriers gave a different answer run
to run; with aliasing off, every seed was stable.
"""
function run!(::Aliasing, c)
    a = analysis(c)
    ts = transients(c)
    for (i, y) in enumerate(ts), (j, x) in enumerate(ts)
        i == j && continue
        arena(x) == arena(y) || continue   # offsets in different allocations never overlap
        x.last < y.first || continue
        a.offsets[i] < a.offsets[j] + nbytes(x) &&
            a.offsets[j] < a.offsets[i] + nbytes(y) || continue
        push!(get!(() -> Tuple{Int,Int}[], a.alias_begins, y.first), (i, j))
    end
    return c
end

"""Transitions per pass, and which of them still need a barrier emitted."""
struct Barriers <: Phase end

"""Resolve and build every pipeline a draw needs, once."""
struct Pipelines <: Phase end

"""
The order phases run in.

Schedule must precede Liveness, because reordering passes changes the intervals.
Place must precede Aliasing, which produces the offsets it needs. Aliasing must
precede Barriers, which has to emit for the hazard it finds.
"""
const PHASES = (Dag(), Schedule(), Liveness(), Place(), Aliasing(), Barriers(), Pipelines())

"""Run the whole pipeline, or a prefix of it when testing one phase in isolation."""
const EMPTY_HANDOVER = Tuple{Int,Int}[]

"""
Walk the passes in order carrying per-resource state. What a pass needs before it
runs is what `transition!` appends while its usages are replayed, so a pass
touching only resources nobody else touched needs nothing and can overlap
whatever ran before it.

Nothing is coalesced across resources: a scoped barrier orders one resource's
memory and nothing else, so a barrier emitted for A never stands in for a hazard
on B however wide its masks. What a pass needs IS its local hazard set, and the
backend lowers it (`passbarriers`) — this phase is the same on every backend,
asking `needs_transition` of the device's [`syncbackend`](@ref).
"""
function run!(::Barriers, c)
    g = c.graph
    be = syncbackend(g.dev)
    states = Dict{Int,ResourceState}()

    # A resource's state at the start of a replay is not "unknown" for anything
    # that outlives the frame: the window comes back from present, an offscreen
    # target from whatever last read it, and every other resource from whatever
    # this same plan did to it last time round.
    #
    # That last one has to be seeded from the *end* of the schedule, because a
    # plan is replayed: frame N+1's first use of a resource follows frame N's last
    # use of it, and with frames in flight the two overlap on the GPU. Seeding
    # `Undefined` instead left the first barrier of a frame with no source scope —
    # nothing to wait for — which synchronization validation reports as a
    # write-after-write against the previous frame's store op, and which is a race
    # the moment the frame loop stops flushing.
    #
    # The layout is unaffected: a discarding destination still transitions from
    # UNDEFINED (see `ImageBarrier`). This is only about what the barrier waits on.
    # Atomic segments, so a partial overlap is tracked rather than refused.
    #
    # This is what VVL's `AccessMap` does incrementally (`layers/sync/sync_access_map.h`:
    # `Split` at each range bound, then `InfillGaps`), done once instead: by the
    # time a plan compiles, every range that will ever be declared is known, so
    # the cuts can be taken up front and the walk left alone. Each usage then
    # stands for the segments its range covers, and a whole-resource usage stands
    # for all of them — after which two usages either name the same segment or do
    # not, which is the only question the per-resource walk knows how to answer.
    slices = sliceindex(g)
    segs = Dict{Int,Vector{Int}}()
    let byparent = Dict{Int,Vector{UnitRange{Int}}}()
        for ((pid, r), _) in g.views
            push!(get!(byparent, pid, UnitRange{Int}[]), r)
        end
        for (pid, ranges) in byparent
            parent = g.ids.by_id[pid]
            n = length(parent)
            cuts = sort!(unique!(vcat([1, n + 1], first.(ranges), last.(ranges) .+ 1)))
            spans = [cuts[k]:(cuts[k + 1] - 1) for k in 1:(length(cuts) - 1)]
            filter!(!isempty, spans)
            ids = map(spans) do s
                resourceid(g, get!(() -> BufferRange(parent, s), g.views, (pid, s)))
            end
            segs[pid] = ids                      # the whole buffer is every segment
            for r in ranges
                sid = resourceid(g, g.views[(pid, r)])
                segs[sid] = [ids[k] for (k, s) in enumerate(spans) if first(s) >= first(r) &&
                                                                      last(s) <= last(r)]
            end
        end
    end
    segments_of(id) = get(segs, id, (id,))

    final = Dict{Int,Type}()
    for p in ordered(c), (id, U) in p.usages, sid in segments_of(id)
        final[sid] = U
    end

    state(id) = get!(states, id) do
        r = g.ids.by_id[id]
        u = get(final, id, nothing)
        u === nothing && (u = initial_state(r))
        u === nothing ? ResourceState(resourcekind(r)) :
                        ResourceState(resourcekind(r), u)
    end

    for (i, p) in enumerate(ordered(c))
        pre = Transition[]
        for (id, U) in p.usages, sid in segments_of(id)
            transition!(pre, be, sid, state(sid), U)
        end

        # A layout change is per image, so no barrier on another resource can have
        # performed it however wide its masks were. Only the memory dependency is
        # coalescable; an image transition dropped here never becomes an
        # `ImageBarrier` below, and the copy after a second colour attachment then
        # reads it in COLOR_ATTACHMENT_OPTIMAL.
        # Coalescing across resources is what a global barrier bought, and it is
        # exactly what a scoped one cannot: a buffer barrier orders that buffer's
        # memory and nothing else, so a barrier emitted for resource A never
        # stands in for a hazard on B however wide its masks are. Dropping one on
        # that reasoning is a race — it is how the showcase's `sun cull` lost the
        # barrier between the clear of its counter and the atomic that reads it.
        #
        # Nothing is left to remove per resource either: `transition!` emits only
        # where a resource's own state actually changes, so the walk is already
        # minimal and `pre` *is* the local hazard set. Kept as a flag because
        # `bench/` still compiles both ways to measure what the old strategy did.
        needed = copy(pre)

        # The barrier where bytes change hands, derived like every other one.
        #
        # It is the one transition not produced by `transition!`, because the
        # hazard is between two resources that never mention each other: no
        # per-resource sequence can see that a different transient was living in
        # this memory. What the compiler *does* know is which ones vacated it —
        # the placer worked that out — and what each was last doing, which is the
        # state the walk above is carrying right now. So it waits for exactly
        # those usages, at exactly the new tenant's first one.
        #
        # It used to name every stage and both access directions instead. That is
        # correct and says nothing: a barrier that waits for everything cannot be
        # wrong, and cannot be checked either.
        handovers = get(analysis(c).alias_begins, i, EMPTY_HANDOVER)
        for newi in unique(first(h) for h in handovers)
            to = usage_of(p, resourceid(g, c.graph.transients[newi]), g)
            to === nothing && continue
            waits = Type[]
            for (a, o) in handovers
                a == newi || continue
                for u in lastuses(states, slices, resourceid(g, c.graph.transients[o]))
                    u in waits || push!(waits, u)
                end
            end
            isempty(waits) && continue
            push!(needed, Transition(0, first(waits), waits, to))
        end

        c.prepass[p] = needed
        append!(c.transitions, pre)
    end
    c
end

function compile!(ctx, phases = PHASES)
    for p in phases
        run!(p, ctx)
    end
    ctx
end
