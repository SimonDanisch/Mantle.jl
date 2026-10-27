# Compilation is a fixed list of phases over a shared context. Each phase runs on
# a context holding only what earlier phases produced, so each is testable alone.

abstract type Phase end

"""
    run!(phase, ctx)

Advance compilation by one phase.
"""
function run!(phase::Phase, ctx)
    throw(MethodError(run!, (phase, ctx)))
end

# The context protocol. All phases live in core so no backend copies the scheduler.

"""Everything the backend-independent phases compute. Returned by [`analysis`](@ref),
so a new phase output adds no per-backend accessor."""
mutable struct Analysis
    deps::Vector{Vector{Int}}                   # Dag
    order::Vector{Int}                          # Schedule
    items::Vector{Item}                         # Liveness
    peak::Int                                   # Place
    naive::Int
    offsets::Vector{Int}
    # Aliasing: pass index => (new, old) transient pairs whose bytes change hands
    # there. The barrier waits on the old transient's last usages.
    alias_begins::Dict{Int,Vector{Tuple{Int,Int}}}
    # Place: the shared region of each arena this plan was placed into, and its key.
    # Not owned: teardown untenants and the last tenant releases the bytes.
    regions::Vector{Any}
    arenas::Vector{Any}
    # Place: per transient, its arena key (its kind, or an `ArenaPart` of it).
    parts::Vector{Any}
end
Analysis() = Analysis(Vector{Int}[], Int[], Item[], 0, 0, Int[],
                      Dict{Int,Vector{Tuple{Int,Int}}}(), Any[], Any[], Any[])

"""
    Compilation

Supertype of a backend's compilation context. The default accessors read fields:
a subtype has `analysis`, `alias`, `policy` and `graph` fields, and the graph has
`passes`, `transients`, `transient_by_id` and `dev`; a differently shaped context
overrides the accessor. A backend always supplies `usages`, `overlapping`,
`overlapgroup` and `pool`.
"""
abstract type Compilation end

"""The [`Analysis`](@ref) a compilation context carries."""
function analysis end
analysis(c::Compilation) = c.analysis

"""The passes a context is compiling, in declaration order."""
function passes end
passes(c::Compilation) = c.graph.passes

"""A pass's declared resource usages, as `id => Usage` pairs."""
function usages end

"""
    overlapping(ctx, a::Int, b::Int) -> Bool

Whether two resource ids can name the same bytes: equal ids, a slice and its
parent, two intersecting slices of one parent. A wrong `false` lets the scheduler
reorder two writes to the same memory; a backend without slices answers `a == b`.
"""
function overlapping end

"""
    overlapgroup(ctx, id::Int) -> Int

A key shared by every id that can overlap `id` (a slice's parent id). Ids in
different groups never overlap, so `Dag` compares a pass only with earlier passes
of the same group; [`overlapping`](@ref) decides within a group.
"""
function overlapgroup end

"""The scheduling [`Policy`](@ref) a context was built with."""
function policy end
policy(c::Compilation) = c.policy

"""The device a context compiles for, which owns its [`Pool`](@ref)."""
function device end
device(c::Compilation) = c.graph.dev

"""The device's [`Pool`](@ref), owned by the device, never by the caller. `Place`
suballocates from it instead of allocating per compile."""
function pool end

"""
    blocksize(dev) -> Int

Smallest block the pool takes from `dev`. 64 MiB suits discrete VRAM, where
allocations are expensive and count-limited; for host memory it over-allocates.
"""
blocksize(dev) = 64 << 20

"""Whether transients may share bytes. `false` gives every transient the whole
timeline, which isolates aliasing hazards when bisecting."""
function alias end
alias(c::Compilation) = c.alias

"""Every transient the graph declared, in declaration order."""
function transients end
transients(c::Compilation) = c.graph.transients

"""`id => transient` for the ids passes name them by."""
function transientbyid end
transientbyid(c::Compilation) = c.graph.transient_by_id

# Transient accessors, answered per backend transient type; phases never look inside.
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

Give a placed transient its storage at `offset` in `slab`. `device` is the
graph's, which owns the transient's storage; with several devices the process
default may be a different one.
"""
function materialize! end

"""
    deviceslice(device, T, dims, memory, offset) -> array

`memory` at `offset` as a `dims`-shaped array of `T`. Borrowed: the pool owns the
memory and this never frees it. All [`materialize!`](@ref) asks of a
KernelAbstractions backend; a backend that keeps the block because its barriers
need the buffer identity implements no method.
"""
function deviceslice end

"""Passes in execution order: the scheduled order once `Schedule` has run,
declaration order before."""
ordered(c) = (o = analysis(c).order; isempty(o) ? passes(c) : passes(c)[o])

"""Which pass depends on which: pass j depends on an earlier pass i when they touch
overlapping resources and at least one writes. Read-after-read is no edge."""
struct Dag <: Phase end

function run!(::Dag, c)
    ps = passes(c)
    a = analysis(c)
    a.deps = [Int[] for _ in ps]
    # Earlier passes per `overlapgroup`: all that touched it, and those that
    # wrote it. A read depends on earlier writers, a write on every earlier pass,
    # of an overlapping id; this avoids comparing every pair of passes.
    touched = Dict{Int,Vector{Tuple{Int,Int}}}()
    written = Dict{Int,Vector{Tuple{Int,Int}}}()
    # `stamp[i] == j` once pass `i` is among `j`'s dependencies.
    stamp = zeros(Int, length(ps))
    none = Tuple{Int,Int}[]
    for j in eachindex(ps)
        deps = a.deps[j]
        for (idj, Uj) in usages(ps[j])
            for (i, idi) in get(writes(Uj) ? touched : written, overlapgroup(c, idj), none)
                stamp[i] == j && continue
                overlapping(c, idi, idj) || continue
                stamp[i] = j
                push!(deps, i)
            end
        end
        sort!(deps)
        for (idj, Uj) in usages(ps[j])
            k = overlapgroup(c, idj)
            push!(get!(Vector{Tuple{Int,Int}}, touched, k), (j, idj))
            writes(Uj) && push!(get!(Vector{Tuple{Int,Int}}, written, k), (j, idj))
        end
    end
    return c
end

"""Choose an execution order consistent with the DAG. A [`Policy`](@ref) sets the
bit position of each objective in the packed score."""
struct Schedule <: Phase end

abstract type Policy end

"""Prefer passes that free more bytes than they allocate: smaller peak, more
ordering between passes."""
struct Compact <: Policy end

"""Prefer declaration order and leave lifetimes long: larger peak, fewer
dependencies forced by shared memory. The default."""
struct Overlap <: Policy end

# The same global memory dependency covers prior accesses to every buffer, not
# just the resource whose hazard caused it. Preserve exact usage keys rather
# than guessing that a read-only execution dependency also flushes writes.
replay_hazard_key(t::Transition) = (Tuple(sort(t.waits; by = string)), t.to)

"""Bit offset of each objective in the packed score; a policy is a choice of offsets."""
memory_shift(::Compact) = 16
memory_shift(::Overlap) = 0
order_shift(::Compact) = 0
order_shift(::Overlap) = 16

"""Maximum of each of the two memory terms, so their sum fits a 16-bit field."""
const SCORE_MAX = 0x1fff

"""Greedy list scheduling. Objectives occupy disjoint bit ranges of one integer score,
so candidates compare with one integer compare. The memory term counts a transient's
first touch as an allocation and its last as a free."""
function run!(::Schedule, c)
    ps = passes(c)
    n = length(ps)
    n == 0 && return c

    deps = loopdeps(ps, analysis(c).deps)
    remaining = [length(d) for d in deps]
    dependents = [Int[] for _ in 1:n]
    for j in 1:n, i in deps[j]
        push!(dependents[i], j)
    end

    # Passes touching each resource, so its last use is known. Counting usages
    # would double-count a pass that binds a resource twice and hide the last use.
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
    # A loop body is scheduled contiguously: once its first pass is taken, only
    # its passes are candidates until the last. `left` counts each loop's remainder.
    left = IdDict{Any,Int}()
    for p in ps
        p.loop === nothing || (left[p.loop] = get(left, p.loop, 0) + 1)
    end
    inloop = nothing

    while !isempty(ready)
        best, bestscore = 0, -1
        for k in eachindex(ready)
            i = ready[k]
            inloop === nothing || ps[i].loop === inloop || continue
            alloc = 0
            freed = 0
            for id in ids(ps[i])
                b = get(bytesof, id, 0)
                b == 0 && continue
                get(seen, id, 0) == 0 && (alloc += b)
                get(seen, id, 0) + 1 == touches[id] && (freed += b)
            end
            # Two terms, not a difference clamped at zero, which would score break-even
            # and net-allocating passes alike. Scaled by the largest per-pass
            # allocation, not a fixed shift, so small buffers do not round to zero.
            mem = clamp((SCORE_MAX * (maxalloc - alloc)) ÷ maxalloc, 0, SCORE_MAX) +
                  clamp((SCORE_MAX * freed) ÷ maxalloc, 0, SCORE_MAX)
            # Clamped because the fields are OR'd: a term spilling its 16 bits
            # corrupts the one above it.
            ord = clamp(n - i, 0, 0xffff)                # earlier declaration scores higher
            score = (mem << mshift) | (ord << oshift)
            if score > bestscore
                best, bestscore = k, score
            end
        end
        best == 0 && error("scheduling: a loop body was entered and none of its " *
                           "remaining passes is ready, which `loopdeps` rules out")
        i = ready[best]
        deleteat!(ready, best)
        push!(order, i)
        if ps[i].loop !== nothing
            l = ps[i].loop
            left[l] -= 1
            inloop = left[l] == 0 ? nothing : l
        end
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

"""
    loopdeps(passes, deps) -> deps

The Dag with every [`repeat!`](@ref) body enterable only as a whole: each body
pass depends on everything outside the loop that any body pass depends on. The
scheduler keeps a body contiguous once entered, which needs this guarantee.
"""
function loopdeps(ps, deps)
    any(p -> p.loop !== nothing, ps) || return deps
    deps = [copy(d) for d in deps]
    members = IdDict{Any,Vector{Int}}()
    for (i, p) in enumerate(ps)
        p.loop === nothing || push!(get!(members, p.loop, Int[]), i)
    end
    for (_, ms) in members
        inside = Set(ms)
        ext = unique!([d for m in ms for d in deps[m] if !(d in inside)])
        for m in ms
            for d in ext
                d in deps[m] || push!(deps[m], d)
            end
        end
    end
    return deps
end

"""
    loopspans(passes) -> Vector{Tuple{UnitRange{Int},Loop}}

The loops among scheduled `passes`, as the contiguous range each body occupies.
Throws if a body is not contiguous.
"""
function loopspans(ps)
    out = Tuple{UnitRange{Int},Loop}[]
    i = 1
    while i <= length(ps)
        l = ps[i].loop
        if l === nothing
            i += 1
            continue
        end
        j = i
        while j < length(ps) && ps[j + 1].loop === l
            j += 1
        end
        any(x -> x[2] === l, out) && error(
            "scheduling split the body of a loop; it has to be one run of passes")
        push!(out, (i:j, l))
        i = j + 1
    end
    return out
end

"""Which passes a resource is live across. Inputs to placement."""
struct Liveness <: Phase end

"""The transient usage `id` belongs to, through a slice or view, or `nothing`. A
`range` usage or [`ResourceView`](@ref) is interned as a `BufferRange` with its own
id, but the interval belongs to the parent: touching any part keeps all of it live."""
function transientfor(c, byid, id)
    t = get(byid, id, nothing)
    t === nothing || return t
    r = get(c.graph.ids.by_id, id, nothing)
    r isa BufferRange || return nothing
    return get(byid, get(c.graph.ids, r.parent, 0), nothing)
end

"""Intervals come from the passes that use each transient, in scheduled order. With
`alias = false` every transient spans the whole timeline. Throws for an unused transient."""
function run!(::Liveness, c)
    ts = transients(c)
    isempty(ts) && return c
    byid = transientbyid(c)
    # From the scheduled order: reordering passes changes the intervals.
    for t in ts
        t.first, t.last = typemax(Int), 0
    end
    ord = ordered(c)
    for (pos, p) in enumerate(ord), (id, _) in usages(p)
        t = transientfor(c, byid, id)
        t === nothing && continue
        t.first = min(t.first, pos)
        t.last = max(t.last, pos)
    end
    # A loop body runs again after itself. A transient shared with passes outside
    # the body, or first read in it (carrying the previous iteration's value), is
    # live across the whole body. One written before it is read is per iteration;
    # the full barrier at each iteration's head orders its reuse.
    for (r, _) in loopspans(ord)
        seen = Base.IdSet{Any}()
        carried = Base.IdSet{Any}()
        for pos in r
            firsthere = Base.IdSet{Any}()
            for (id, U) in usages(ord[pos])
                t = transientfor(c, byid, id)
                t === nothing && continue
                (t in seen && !(t in firsthere)) && continue
                push!(seen, t); push!(firsthere, t)
                reads(U) && push!(carried, t)
            end
        end
        for t in seen
            if t in carried || t.first < first(r) || t.last > last(r)
                t.first = min(t.first, first(r))
                t.last = max(t.last, last(r))
            end
        end
    end
    # Otherwise the placer reports the missing interval as an inverted `Span`.
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

"""Offsets for every transient, from the liveness intervals. Not `Placement`,
which is the result of `place`."""
struct Place <: Phase end

"""Assign offsets, one placement per arena. Peak sums over arenas, which are
separate allocations."""
function run!(::Place, c)
    a = analysis(c)
    isempty(a.items) && return c
    ts = transients(c)
    a.offsets = zeros(Int, length(ts))
    a.parts = Vector{Any}(undef, length(ts))
    a.peak = 0
    a.naive = sum(nbytes, ts)
    for ar in unique(map(arena, ts))
        # One allocation per part. Items placed entirely inside the room keep their
        # offsets (a subset of a valid placement is valid); the rest is placed again
        # as the next part. Each part asks `arenaroom` after the previous one
        # reserved, so all parts together stay within the device budget.
        rest = findall(t -> arena(t) == ar, ts)
        part = 0
        while !isempty(rest)
            part += 1
            key = arenapart(ar, part)
            # Checked here, not at `rawalloc`: only here are the items known, so
            # the error can name them instead of blaming the next allocation.
            prob = Problem(a.items[rest], arenaroom(pool(c), device(c), key))
            pl = place(prob)
            if pl.height > prob.capacity
                # The headroom counts every pool block as spent, including finished
                # arenas, unswept staging and unfinalized buffers. Allocation
                # recovers these before failing (`acquire!`), so the bound must too.
                makeroom!(pool(c), device(c))
                prob = Problem(a.items[rest], arenaroom(pool(c), device(c), key))
                pl = place(prob)
            end
            ends(i) = pl.offsets[string(i)] + a.items[i].size
            idx = pl.height <= prob.capacity ? rest : filter(i -> ends(i) <= prob.capacity, rest)
            # Nothing fits: an item exceeds one allocation or the budget is spent.
            isempty(idx) && checkcapacity(prob, pl, "arena $key")
            height = maximum(ends, idx)
            # `reserve!`, not `acquire!`: every plan in an arena gets the same
            # region, so the arena costs the largest plan, not the sum. The region
            # must be aligned as strictly as any item, since item offsets are
            # relative to it (a Metal render target needs 2048, above the 256 default).
            want_align = maximum(i -> a.items[i].alignment, idx; init = REGION_ALIGN)
            want_align = max(want_align, REGION_ALIGN)
            reg = reserve!(pool(c), device(c), key, ts[idx], height;
                           align = want_align, blocksize = blocksize(device(c)))
            push!(a.regions, reg)
            push!(a.arenas, key)
            a.peak += height
            for i in idx
                a.offsets[i] = pl.offsets[string(i)]
                a.parts[i] = key
                # Item offsets are relative to the region, the region's to the block.
                materialize!(device(c), ts[i], memoryof(reg), offset(reg) + a.offsets[i])
            end
            rest = setdiff(rest, idx)
        end
    end
    return c
end

"""
    giveup!(pool, dev, regions, arenas, x)

Untenant `x` from the arenas it was placed into and empty both vectors. Explicit,
never a finalizer (see [`trim!`](@ref)). Regions are shared, so the last tenant
out releases the bytes; a plan dropped without this keeps its arena at size until
the pool is trimmed. Takes the vectors because a compiled plan keeps no `Analysis`.
"""
function giveup!(pool, dev, regions, arenas, x)
    for ar in arenas
        untenant!(pool, dev, ar, x)
    end
    empty!(regions)
    empty!(arenas)
    return nothing
end

"""Which passes begin using memory another transient just finished with. Per-resource
tracking cannot see this hazard because two aliased transients look unrelated."""
struct Aliasing <: Phase end

"""Fill `alias_begins`: each pass where a transient starts on bytes of a dead earlier
one, whose first write would otherwise race the old tenant's last access."""
function run!(::Aliasing, c)
    a = analysis(c)
    ts = transients(c)
    # Per allocation, sorted by offset: a transient's bytes meet only those that
    # start inside it. Offsets in different allocations (another arena, or another
    # part `Place` spilled) never overlap.
    byarena = Dict{Any,Vector{Int}}()
    for i in eachindex(ts)
        push!(get!(Vector{Int}, byarena, a.parts[i]), i)
    end
    for (_, idx) in byarena
        sort!(idx; by = i -> a.offsets[i])
        # Two loops and not `for p in …, q in …`: a `break` there leaves both.
        for (p, i) in enumerate(idx)
            for q in (p + 1):length(idx)
                j = idx[q]
                a.offsets[j] < a.offsets[i] + nbytes(ts[i]) || break
                aliasbegin!(a, ts, i, j)
                aliasbegin!(a, ts, j, i)
            end
        end
    end
    # Sorted: barriers are emitted in this order, which must not follow Dict order.
    foreach(sort!, values(a.alias_begins))
    return c
end

"""Transient `i` begins where `j`'s bytes were, after `j` is dead: a handover."""
function aliasbegin!(a, ts, i::Int, j::Int)
    y, x = ts[i], ts[j]
    x.last < y.first || return nothing
    a.offsets[i] < a.offsets[j] + nbytes(x) && a.offsets[j] < a.offsets[i] + nbytes(y) ||
        return nothing
    push!(get!(() -> Tuple{Int,Int}[], a.alias_begins, y.first), (i, j))
    return nothing
end

"""Transitions per pass, and which of them still need a barrier emitted."""
struct Barriers <: Phase end

# A dispatch's arguments are simultaneous accesses, not a sequence of GPU
# commands. In particular `(out, (out,))` declares write and read separately.
# Advancing state through those two entries used to lose the write before the
# following pass. Join storage directions after expanding overlapping ranges.
join_pass_storage(::Type,::Type)=nothing
join_pass_storage(::Type{Storage{K,A}},::Type{Storage{K,B}}) where {K,A,B} =
    Storage{K,Access{reads(A)||reads(B),writes(A)||writes(B)}}

function barrier_pass_usages(p,segments_of)
    out=Pair{Int,Type}[]
    last=Dict{Int,Int}()
    for (id,U) in p.usages,sid in segments_of(id)
        at=get(last,sid,0)
        merged=at==0 ? nothing : join_pass_storage(out[at].second,U)
        if merged===nothing
            push!(out,sid=>U);last[sid]=length(out)
        else
            out[at]=sid=>merged
        end
    end
    out
end

"""Resolve and build every pipeline a draw needs, once."""
struct Pipelines <: Phase end

"""The order phases run in. `Schedule` precedes `Liveness` (reordering changes
intervals), `Place` precedes `Aliasing` (which needs offsets), and `Aliasing`
precedes `Barriers` (which emits for the hazards it finds)."""
const PHASES = (Dag(), Schedule(), Liveness(), Place(), Aliasing(), Barriers(), Pipelines())

const EMPTY_HANDOVER = Tuple{Int,Int}[]

"""Walk the passes in scheduled order carrying per-resource state; a pass needs the
transitions `transition!` appends while its usages replay. Shared by every backend
through the device's [`syncbackend`](@ref); `passbarriers` lowers the result."""
function run!(::Barriers, c)
    g = c.graph
    be = syncbackend(g.dev)
    replay=ordered_entry(c)
    states = Dict{Int,ResourceState}()
    coalesce=replay&&global_buffer_barriers(be)
    visible=Dict{Any,Int}()
    lasttouch=Dict{Int,Int}()

    # A resource starts in the state of its last usage in this schedule: frame
    # N+1's first use follows frame N's last and may overlap it in flight. Seeding
    # `Undefined` gives the first barrier no source scope, a write-after-write race.
    #
    # Ranged buffers are cut into segments at every declared range bound, so
    # partial overlaps are tracked. A usage stands for the segments its range
    # covers, a whole-resource usage for all of them.
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

    passes=ordered(c)
    usages=[barrier_pass_usages(p,segments_of) for p in passes]
    final = Dict{Int,Type}()
    for pu in usages, (sid,U) in pu
        final[sid] = U
    end

    # A replayed plan starts each segment where the previous run left it,
    # unless the recording's full entry dependency has already ordered that use.
    state(id) = get!(states,id) do
        kind=resourcekind(g.ids.by_id[id])
        replay&&kind isa BufferKind ? ResourceState(kind) : ResourceState(kind,final[id])
    end

    for (i, p) in enumerate(passes)
        pre = Transition[]
        alias_last=Dict{Int,Int}()
        for (sid,U) in usages[i]
            # A recorded update pass emits no stores: they are submitted in
            # front, then covered by the recording's full head dependency.
            replay&&p.kind===:update&&resourcekind(g.ids.by_id[sid]) isa BufferKind&&continue
            transition!(pre, be, sid, state(sid), U)
        end

        # The portable answer is per-resource. Native GLOBAL
        # buffer dependencies can coalesce covered hazards below; images still
        # require their own layout transitions.
        needed = pre

        # The aliasing handover, which `transition!` cannot see because the two
        # transients never name each other. It waits for the last usages of the
        # transients that vacated the memory, at the new tenant's first usage.
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
            if coalesce
                alias_last[length(needed)]=maximum((get(lasttouch,sid,0)
                    for (a,o) in handovers if a==newi
                    for sid in segments_of(resourceid(g,c.graph.transients[o])));init=0)
            end
        end

        if coalesce
            kept=Transition[]
            for (j,t) in enumerate(needed)
                # Image barriers still carry per-image layout/state. Only
                # buffers and arena-memory handovers have global cache scope.
                buffer=t.resource==0||resourcekind(g.ids.by_id[t.resource]) isa BufferKind
                source=t.resource==0 ? alias_last[j] : get(lasttouch,t.resource,0)
                if !buffer||source>get(visible,replay_hazard_key(t),0)
                    push!(kept,t)
                end
            end
            # Barriers precede this pass, so they cover accesses through i-1,
            # not the writes the pass itself is about to issue.
            for t in kept
                (t.resource==0||resourcekind(g.ids.by_id[t.resource]) isa BufferKind)||continue
                visible[replay_hazard_key(t)]=i-1
            end
            needed=kept
        end

        c.prepass[p] = needed
        if coalesce
            for (sid,_) in usages[i]
                lasttouch[sid]=i
            end
        end
    end
    c
end

"""
    compile!(ctx, phases = PHASES) -> ctx

Run `phases` in order on `ctx`; a prefix of [`PHASES`](@ref) tests one phase in isolation.
"""
function compile!(ctx, phases = PHASES)
    for p in phases
        run!(p, ctx)
    end
    ctx
end
