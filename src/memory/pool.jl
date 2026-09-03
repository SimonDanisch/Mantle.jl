# Suballocation across plans. Backend-independent, on purpose.
#
# `Place` already packs one arena's transients into `height` bytes. This is the
# layer under that: where those `height` bytes come from, and how they are reused
# once the plan that asked for them is gone.
#
# It lives here rather than in a backend for the same reason the phases do — a
# second backend must not need a second suballocator. A backend supplies four
# primitives and no policy: `rawalloc`, `rawfree`, `constraintof` and
# `compatible`. Everything about which block, what offset, when to grow and when
# to release is decided in this file.
#
# It also must not sit on a backend's own allocator. Mantle asking Lava for a
# `LavaArray` and suballocating inside it stacks two allocators, and the lower one
# is invisible to the thing claiming to manage memory. `rawalloc` is meant to be
# the rawest allocation the device offers — `vkAllocateMemory`, not a pooled
# wrapper around it.

"""
    rawalloc(dev, kind, bytes, constraint) -> memory

One raw allocation from the device, of at least `bytes`, legal for `constraint`.

Called once per BLOCK, never per transient and never per compile. A backend that
routes this through its own pool defeats the point: the memory Mantle hands out
would then be a suballocation of a suballocation, and the peak it reports would
be about its own bookkeeping rather than about the device.
"""
function rawalloc end

"""Release a block's allocation. Called only when nothing references it."""
function rawfree end

"""
    maxalloc(dev) -> Int

The largest single allocation this device will make.

A harder limit than [`capacity`](@ref) and usually a far smaller one — 4 GB
against 29 GB of budget on the machine this was written on — and the one that
actually bounds a block, because a block is *one* allocation: the offsets a
placement produces are into a single contiguous range, so it cannot be split
across two.

Measured rather than assumed. The alternative was a reserve fraction subtracted
from the heap, and every such fraction is either too small to prevent the failure
or too large to justify. A plan needing 5 GB in one arena cannot run on this
device however idle the card is, and saying so by name is worth more than a
percentage that happens to catch it here.

`typemax(Int)` is the default: a backend that will not say is not blocked, it is
merely bounded by `capacity` alone.
"""
maxalloc(dev) = typemax(Int)


"""
    constraintof(dev, kind, transients) -> constraint

What memory the given transients can legally share, as whatever the backend needs
to answer [`compatible`](@ref) later — a memory-type mask, a usage flag set.

Derived from the transients rather than declared: on Vulkan an image's memory
type bits are a property of the image, and a block that hosts several must
satisfy all of them at once.
"""
function constraintof end

"""
    mergeconstraints(dev, kind, a, b) -> constraint | nothing

One constraint that satisfies both, or `nothing` when none can.

An arena outlives the plan that first sized it, so the block behind it has to
keep satisfying every tenant placed there — and what "both" means is the
backend's: buffer usage bits UNION (the block must permit everything any tenant
does with it) while image memory-type bits INTERSECT (the type has to be one
every image can bind to). Getting those the wrong way round is memory that works
until the one image whose type was excluded is bound.

The default demands equality, which is the safe reading for a backend that has
not said otherwise: two constraints that are not the same are not reconciled by
core guessing.

Impossibility is a THROW, not a returned `nothing`. `nothing` is a perfectly good
constraint — it is the host backend's, whose memory is just memory — so using it
as the failure signal made every host graph unplaceable. A sentinel that a
backend can legitimately return is not a sentinel.
"""
mergeconstraints(dev, kind, a, b) = a == b ? a : throw(ArgumentError(
    "arena $kind cannot host this plan and the ones already placed in it: it " *
    "needs $b where they need $a, and one allocation cannot serve both. This is " *
    "not a size problem — a bigger arena would not help."))

"""
    compatible(dev, block_constraint, request) -> Bool

May a block created for `block_constraint` host a request needing `request`?

Core cannot answer this — "what memory is legal here" is exactly the backend's
semantics. It is why blocks are not interchangeable and why a pool cannot simply
be one large buffer.
"""
function compatible end

"""One bin per power of two, which is as many as an `Int` length can reach."""
const NBINS = 8 * sizeof(Int)

"""
The unused parts of one block, indexed three ways so that both ends of an
allocation are O(1): by where a span starts, by where it ends, and bucketed by
how large it is.

The obvious structure — one sorted `Vector{Span}`, first fit to carve,
`insert!` and coalesce to release — is what this replaces, and it was adequate
while the pool served only graph arenas: a handful of live regions per block,
one free entry per plan, and `Place` had already packed the transients inside
each one.

What it does not have is a bound. First fit scans the free list from the front,
and `insert!`/`deleteat!` memmove half of it per release, so both are O(number of
free spans) — and once the pool serves general device allocation rather than a
handful of arenas, nothing keeps that number small. Measured, taking the
ordinary shape of many small arrays with half of them freed and staying freed,
then a run of larger allocations no hole can host (2000 of them, this machine):

    scattered holes    vector     bins
              250     0.76 ms   0.85 ms
              500     1.47      1.02
             1000     2.86      1.38
             2000     5.67      2.06
             4000    11.16      3.53
             8000    22.21      6.72

The vector is exactly linear in the holes it has to get past. The point is not
the 3.3x at the right-hand end — it is that the right-hand end keeps going.

**Below about 350 free spans the vector is the faster of the two**, by around
10%, because a `Dict` operation costs more than a `Vector` one and at that size
neither is scanning far enough for it to matter. That is the price paid here, and
it is paid deliberately: a 10% constant against an unbounded worst case. A churn
benchmark that reuses its holes immediately never leaves the regime where the
vector wins — so this is a bound being bought, not a speedup being claimed.

Everything here is O(1) amortised instead:

  * `bylower` and `byupper` turn coalescing into two dictionary lookups, where
    the vector needed a binary search and a shift.
  * `bins[k]` holds the LOWER of every free span whose length falls in
    `[2^(k-1), 2^k)`, so a fit is found by taking an entry out of a high enough
    bin rather than by searching for one.
  * `slot` records where each span sits inside its bin, which is what makes
    removing it a swap-with-last instead of a scan.

Scanning bins upward from the request's own bin also changes the fit policy
from first to approximate-best, which is the better default here: it leaves the
large spans intact for the requests that need them.
"""
struct FreeSpans
    bylower::Dict{Int,Int}          # span start  => span end
    byupper::Dict{Int,Int}          # span end    => span start
    bins::Vector{Vector{Int}}       # size class  => starts of the spans in it
    slot::Dict{Int,Int}             # span start  => its index within its bin
end
FreeSpans() = FreeSpans(Dict{Int,Int}(), Dict{Int,Int}(),
                        [Int[] for _ in 1:NBINS], Dict{Int,Int}())

"""The bin holding spans of length `n`: bin `k` is `[2^(k-1), 2^k)`."""
@inline binof(n::Int) = n <= 0 ? 1 : NBINS - leading_zeros(n)

"""
The lowest bin every one of whose spans is at least `n` long.

The point of the distinction: from `binceil(n)` upward, *any* entry fits and can
be taken without looking at it. `binof(n)` itself straddles `n`, so its entries
have to be checked — which is what the bounded scan in [`carve!`](@ref) does.
"""
@inline binceil(n::Int) = n <= 1 ? 1 : binof(n - 1) + 1

@inline function addspan!(fs::FreeSpans, lo::Int, hi::Int)
    hi > lo || return fs                       # an empty span is not free space
    k = binof(hi - lo)
    push!(fs.bins[k], lo)
    fs.slot[lo] = length(fs.bins[k])
    fs.bylower[lo] = hi
    fs.byupper[hi] = lo
    return fs
end

@inline function dropspan!(fs::FreeSpans, lo::Int)
    hi = fs.bylower[lo]
    v = fs.bins[binof(hi - lo)]
    i = fs.slot[lo]
    # Swap-with-last: the bin is a bag, so order carries no meaning and moving
    # the tail entry into the hole costs one slot update instead of a memmove.
    last = v[end]
    v[i] = last
    fs.slot[last] = i
    pop!(v)
    delete!(fs.slot, lo)
    delete!(fs.bylower, lo)
    delete!(fs.byupper, hi)
    return hi
end

"""
One raw allocation, which parts of it are unused, and which parts are out on
loan.

`live` is the exact set of regions [`carve!`](@ref) has handed out and
[`release!`](@ref) has not taken back — offset to end, one entry each. It is
what makes a bad release an error rather than corruption, and it is a stronger
check than the free list can give on its own: with spans coalescing, a region
that has already been merged into a neighbour leaves no trace in the free list
to recognise, so releasing it twice would quietly hand the same bytes out again.

`lock` is per block rather than per pool because [`release!`](@ref) takes no
pool: a `Region` knows only its block. The order, wherever both are held, is
pool then block, and never the reverse.
"""
mutable struct Block
    memory::Any
    bytes::Int
    kind::Any
    constraint::Any
    free::FreeSpans
    live::Dict{Int,Int}
    lock::ReentrantLock
end
Block(memory, bytes::Int, kind, constraint) =
    Block(memory, bytes, kind, constraint,
          addspan!(FreeSpans(), 0, bytes), Dict{Int,Int}(), ReentrantLock())

"""
A slice of a block, held by whoever asked for it.

`offset` is what `materialize!` needs — a transient's own offset from `Place` is
relative to the region, and the two add.
"""
struct Region
    block::Block
    span::Span
end
offset(r::Region) = r.span.lower
Base.length(r::Region) = length(r.span)
memoryof(r::Region) = r.block.memory

"""
One arena's shared bytes, and the plans placed in them.

**Every tenant starts at offset 0**, so two plans on one device alias each other
by construction and the arena is sized to the LARGEST of them rather than to
their total. That is the whole reason a device owns its arenas: two plans that
each acquired their own region could never share, whatever the placer did inside
either one. Measured on SAM 2 and MatAnyone, whose scratch never coexists.

Two things make the overlap safe, and neither is here. Ordering is one barrier at
the head of a recording when the arena has more than one live tenant — the
backend emits it, because only the backend has a command stream. Data lifetime is
the caller's: a plan's transients are gone the moment another tenant runs, which
is the same rule that already governs two runs of one plan.

`tenants` is weak. A plan nobody holds is a plan nobody can run, so it neither
keeps its graph alive nor needs remapping.
"""
mutable struct Arena
    region::Union{Nothing,Region}
    bytes::Int
    constraint::Any        # what every tenant placed here needs, merged
    tenants::Vector{WeakRef}
    lastrun::Any           # who wrote these bytes most recently
end
Arena() = Arena(nothing, 0, nothing, WeakRef[], nothing)

"""Live tenants, pruning collected ones on the way past."""
function tenants!(a::Arena)
    n = 0
    for wr in a.tenants
        wr.value === nothing && continue
        n += 1
        a.tenants[n] = wr
    end
    resize!(a.tenants, n)
    return a.tenants
end

"""
    remap!(tenant, kind, region)

Re-materialise a tenant's transients of arena `kind` into its new region.

`kind` because a tenant is usually placed in more than one arena — buffers and
images are separate allocations — and only one of them moved.

Called only when an arena actually grew. A tenant keeps the offsets its own
placement produced — those are unaffected by the arena moving underneath — so
this rebinds storage and touches nothing the compiler decided.
"""
function remap! end

"""
    remappable(tenant) -> Bool

Whether this tenant can survive its arena moving. `true` unless it says otherwise.

The one thing a tenant knows that the pool cannot: a plan whose commands have
been written once and are submitted again every run (`record!`) holds the
addresses the old region had, and nothing can rewrite a command buffer.
Re-materialising underneath it produces a run that reads freed storage —
deterministic, quiet, and wrong, which is the worst of the three.

So growth asks first and refuses, rather than discovering it later. Asked of
every live tenant, not of the one growing: it is the INCUMBENT that cannot move.
"""
remappable(x) = true

"""
Blocks per arena kind, and the shared arena each kind is placed into.

Growth ADDS a block; it never reallocates an existing one. That is the whole
reason for a list: a plan holding offsets into a block must not have the ground
move under it because another plan needed more room. An ARENA can still outgrow
its region — a later, larger tenant — and that is what `remap!` is for; the block
list is what makes the new region a fresh carve rather than a reallocation of
memory somebody still points into.
"""
struct Pool
    blocks::Dict{Any,Vector{Block}}
    arenas::Dict{Any,Arena}
    # Regions handed back from a context that must not touch a free list, and
    # the same regions once stamped with a fence. See `retire!` / `reclaim!`.
    pending::Vector{Region}
    retiring::Vector{Tuple{Region,Any}}
    lock::ReentrantLock
end
Pool() = Pool(Dict{Any,Vector{Block}}(), Dict{Any,Arena}(),
              Region[], Tuple{Region,Any}[], ReentrantLock())

arenaof(p::Pool, kind) = get!(Arena, p.arenas, kind)

"""
    movable(pool) -> Bool

Whether every live tenant of every arena could survive its memory moving.

`false` means a plan somewhere has been recorded: its command buffer holds the
addresses its regions have today, and nothing can rewrite a command buffer.
`growarena!` asks this of ONE arena before growing it; the block trim asks it of
the whole pool before destroying a buffer, which is the same hazard with no arena
to name it against.

It replaces a `bq.capturing !== nothing` read in the Vulkan backend, which asked
whether a capture happened to be OPEN. That is a different question and it was
answerable only while `bake!` was running — a recording that had already been
taken pinned nothing, so the trim was free to destroy a block it named.
"""
movable(p::Pool) =
    all(a -> all(wr -> remappable(wr.value), tenants!(a)), values(p.arenas))

blocksof(p::Pool, kind) = get!(() -> Block[], p.blocks, kind)

"""Every byte the pool has taken from the device, across all blocks."""
reserved(p::Pool) = sum(b -> b.bytes, Iterators.flatten(values(p.blocks)); init = 0)

"""
    largestfree(pool, kind) -> Int

The biggest span any existing block of `kind` could still hand out.

What makes the capacity bound TIGHT rather than merely safe: a request this size
or smaller reaches no device allocation at all, so bounding it by what the device
has left would refuse a placement that costs nothing. Compatibility is not
consulted — this is a bound, and an over-estimate here can only be corrected by
`acquire!` going on to allocate, which the bound has already been checked against.
"""
function largestfree(p::Pool, kind)
    best = 0
    for b in blocksof(p, kind)
        lock(b.lock) do
            # Only the highest occupied bin can hold the winner: every span in
            # bin k is shorter than every span in bin k+1, by construction.
            for k in NBINS:-1:1
                v = b.free.bins[k]
                isempty(v) && continue
                for lo in v
                    best = max(best, b.free.bylower[lo] - lo)
                end
                break
            end
        end
    end
    return best
end

"""
    headroom(pool, dev, kind) -> Int

How large a single request of `kind` can still be satisfied.

Two bounds and the smaller wins: what the device will hand over in one piece, and
what is left of its budget once everything this pool already holds is subtracted
— plus whatever an existing block could absorb without allocating at all.

In core rather than in a backend because only the pool knows what it has
reserved, and the two device facts it needs are already primitives. The Lava
extension used to compute this itself, against a per-arena allocation it grew by
reallocating; blocks make the arithmetic simpler as well as the memory safer.
"""
headroom(p::Pool, dev, kind) =
    min(maxalloc(dev), max(capacity(dev) - reserved(p), 0) + largestfree(p, kind))

"""First offset at or after `from` that satisfies `align`."""
alignup(from::Int, align::Int) = align <= 1 ? from : cld(from, align) * align

"""
How far into a bin `carve!` looks before giving up on it.

Only bins below `binceil` need looking at at all — from there up, every span is
long enough and the first entry is taken unseen. Those lower bins straddle the
request, so an entry may fail on alignment, and this bounds what that costs.

A miss is not a wrong answer: the search moves to the next bin and eventually to
one where everything fits, so the worst case is a block allocated that a longer
scan would have avoided. Bounded rather than exhaustive because the alternative
reintroduces exactly the O(n) scan this structure exists to remove.
"""
const BIN_SCAN = 8

"""
Carve `bytes` out of `blk`, honouring `align`, or `nothing` if it does not fit.

Approximate best fit: bins are searched upward from the request's own, so a
request takes the smallest span that will hold it and the long spans stay whole
for the requests that need them. The vector this replaced was first fit, which
walks up a block chopping the front off whatever it meets.
"""
function carve!(blk::Block, bytes::Int, align::Int)
    lock(blk.lock) do
        fs = blk.free
        # Worst case the alignment costs `align - 1` bytes of the span, so from
        # this bin up nothing has to be examined.
        guaranteed = binceil(bytes + align - 1)
        for k in binof(bytes):NBINS
            v = fs.bins[k]
            isempty(v) && continue
            if k >= guaranteed
                return takespan!(blk, v[end], bytes, align)
            end
            # This bin straddles the request: check entries, but not all of them.
            for i in length(v):-1:max(1, length(v) - BIN_SCAN + 1)
                lo = v[i]
                start = alignup(lo, align)
                start + bytes <= fs.bylower[lo] &&
                    return takespan!(blk, lo, bytes, align)
            end
        end
        return nothing
    end
end

"""Split the free span at `lo`, hand back the middle, put the ends back."""
function takespan!(blk::Block, lo::Int, bytes::Int, align::Int)
    fs = blk.free
    hi = dropspan!(fs, lo)
    start = alignup(lo, align)
    # Whatever alignment skipped, and whatever is left over, go back — dropping
    # either leaks bytes that nothing can ever hand out again.
    addspan!(fs, lo, start)
    addspan!(fs, start + bytes, hi)
    blk.live[start] = start + bytes
    return Region(blk, Span(start, start + bytes))
end

"""The first block of `kind` that can host `bytes`, or `nothing` if none can."""
function carveany!(pool::Pool, dev, kind, bytes::Int, align::Int, want)
    for blk in blocksof(pool, kind)
        compatible(dev, blk.constraint, want) || continue
        r = carve!(blk, bytes, align)
        r === nothing || return r
    end
    return nothing
end

"""
    acquire!(pool, dev, kind, transients, bytes; align) -> Region

`bytes` from a block that can host `transients`, growing the pool if none can.

A new block is at least `bytes` and at least `blocksize`, so a run of small plans
does not produce a device allocation each. `blocksize` is a kwarg rather than a
constant because the right value is a device property, not a Mantle opinion.
"""
function acquire!(pool::Pool, dev, kind, transients, bytes::Int;
                  align::Int = 256, blocksize::Int = 64 << 20, constraint = nothing)
    bytes = max(bytes, 1)
    # `constraint` given means the caller knows more than these transients do —
    # an arena reconciling what its other tenants also need. Deriving it here
    # would size the block for this plan alone.
    want = constraint === nothing ? constraintof(dev, kind, transients) : constraint
    # Three attempts before the device is asked for anything, and the last two
    # are why `free!` can be deferred without a rebuild doubling peak memory.
    # Bytes a caller just gave back are RETIRED, not free: they are waiting on
    # work the device may already have finished. Growing while they sit there is
    # exactly the growth a pool exists to prevent — so try, take back what the
    # device is done with, try again, wait for what it has already been given,
    # try once more.
    #
    # `wait = true` cannot block for ever: `waitfor` declines a fence whose work
    # has not been submitted, so the worst case is this falls through and grows,
    # which is what it would have done anyway.
    #
    # Under the pool lock for the whole body, `blocksof` included. Two threads
    # allocating used to walk and `push!` the same block vector, which is the
    # `ConcurrencyViolationError` this codebase has already paid for once in a
    # different allocator. `ReentrantLock`, so `reclaim!` below re-entering from
    # this thread is fine — and so is a finalizer landing here, which reaches
    # only `retire!` and touches `pending` rather than any block.
    lock(pool.lock) do
        r = carveany!(pool, dev, kind, bytes, align, want)
        r === nothing || return r
        if reclaim!(pool, dev) > 0
            r = carveany!(pool, dev, kind, bytes, align, want)
            r === nothing || return r
        end
        if reclaim!(pool, dev; wait = true) > 0
            r = carveany!(pool, dev, kind, bytes, align, want)
            r === nothing || return r
        end
        blks = blocksof(pool, kind)
        n = max(bytes, blocksize)
        blk = Block(rawalloc(dev, kind, n, want), n, kind, want)
        push!(blks, blk)
        r = carve!(blk, bytes, align)
        r === nothing && error("a fresh $n-byte block could not host $bytes bytes at " *
                               "alignment $align — the block size or the alignment is wrong")
        return r
    end
end

"""
Give a region back, merging it with any neighbouring free space.

Coalescing here rather than at acquire time is what keeps a plan that recompiles
at the same size from walking up the block: release then acquire finds the span
it just gave back, whole.
"""
function release!(r::Region)
    blk = r.block
    lock(blk.lock) do
        lo, hi = r.span.lower, r.span.upper
        # Checked against what was HANDED OUT, not against the free list. A
        # double release is silent otherwise — the span goes back twice, two
        # later `acquire!`s return the same bytes, and the corruption surfaces
        # nowhere near here. The free list cannot see it either: coalescing
        # dissolves a released span into its neighbours, so by the time the
        # second release arrives there is nothing left that looks like it.
        held = get(blk.live, lo, -1)
        held == hi || error(
            "release!: $(r.span) was not handed out by this block — " *
            (held == -1 ?
             "nothing live starts at $lo, so this region was released already, or it " *
             "outlived the block it came from" :
             "the live region at $lo ends at $held, not $hi"))
        delete!(blk.live, lo)
        # Coalesce with the span ending where this one starts, and the one
        # starting where it ends. Two lookups; at most two joins per release.
        fs = blk.free
        p = get(fs.byupper, lo, -1)
        p == -1 || (dropspan!(fs, p); lo = p)
        haskey(fs.bylower, hi) && (hi = dropspan!(fs, hi))
        addspan!(fs, lo, hi)
    end
    return nothing
end

"""
    fence(dev) -> f

An opaque stand-in for "everything submitted to `dev` up to now".

Opaque because what it is differs per backend and core has no use for the value
beyond handing it back to [`passed`](@ref) — a timeline-semaphore counter on
Vulkan, nothing at all on a backend whose work is synchronous.
"""
function fence end

"""
    passed(dev, f) -> Bool

Has `dev` finished everything that [`fence`](@ref) stood for?

The one question that decides whether a retired region can go back on a free
list, and the reason [`reclaim!`](@ref) is two-phase.
"""
function passed end

"""
    waitfor(dev, f) -> Bool

Block until [`passed`](@ref)`(dev, f)`, and say whether that happened.

**Returns `false` rather than waiting when `f` names work that has not been
submitted yet**, which is the whole subtlety. A fence taken while a command
buffer was open stands for a value only that buffer's submission will signal, and
nothing here can make that happen — an allocator that forced a submit to collect
its own memory would cut a recording in half to do it. So it declines, and the
caller grows the pool instead. Rare, because `run!` reclaims at its head, by
which point the previous frame is submitted.

The counterpart of that rule: this NEVER submits, flushes, or otherwise advances
the device. It only waits for what is already on its way.
"""
function waitfor end

"""
    retire!(pool, r::Region)

Give a region back from a context that must not touch a free list — a finalizer,
which runs on whatever thread the GC picks.

Appends under a lock and does nothing else. [`release!`](@ref) inserts into a
block's free list and coalesces its neighbours; a GC thread doing that while
another `acquire!`s is the bug [`trim!`](@ref) describes below, which Lava has
already paid for once with a `ConcurrencyViolationError` and then a SIGSEGV.

The release itself is [`reclaim!`](@ref), from the owning thread. So this is not
"free from a finalizer" — the finalizer never frees, it only says what is no
longer wanted, and something on the owning thread decides when that is safe.
"""
retire!(p::Pool, r::Region) = (lock(() -> push!(p.pending, r), p.lock); nothing)

"""
    reclaim!(pool, dev) -> Int

Release every retired region the device has finished with, and return how many.

**Two phases, and the delay between them is the point.** A resource nobody holds
cannot be named by work recorded AFTER it was dropped, but work recorded BEFORE
it may still be in flight — and a finalizer is in no position to ask a queue
about that. So the first call that sees a region only stamps it with
[`fence`](@ref), taken here on the owning thread; a later call releases it once
[`passed`](@ref) says the device is through. One full submission boundary between
the drop and the release, with the GC thread never asking the device anything.

Cheap and idempotent, so a renderer can call it every frame; it is a no-op when
nothing has been dropped.

`wait = true` additionally blocks on whatever the device has already been given,
via [`waitfor`](@ref). That is what [`acquire!`](@ref) does before it grows the
pool, and it is the only place in Mantle that waits for the device at all —
which is the point: a caller never has to know what is in flight, because the
one piece of code that needs the bytes does.
"""
function reclaim!(p::Pool, dev; wait::Bool = false)
    # The whole body, not just the `pending` swap. `retiring` was reached
    # unlocked on the assumption that only the owning thread gets here, and
    # `acquire!` — which is now on every allocating thread — calls this.
    # Re-entrant, so `acquire!` already holding it costs nothing.
    lock(p.lock) do
        if !isempty(p.pending)
            f = fence(dev)
            for r in p.pending
                push!(p.retiring, (r, f))
            end
            empty!(p.pending)
        end
        freed = keep = 0
        for (r, f) in p.retiring
            if passed(dev, f) || (wait && waitfor(dev, f))
                release!(r)
                freed += 1
            else
                keep += 1
                p.retiring[keep] = (r, f)
            end
        end
        resize!(p.retiring, keep)
        return freed
    end
end

"""
Hand every fully-unused block back to the device.

Explicit, and never from a finalizer. Lava already learned this one: its pool
free-list was pushed from the GC finalizer thread while another thread popped,
which gave a `ConcurrencyViolationError` and then a SIGSEGV. A pool that frees on
a finalizer is the same bug waiting for a different day.
"""
function trim!(pool::Pool, dev)
    lock(pool.lock) do
        for (kind, blks) in pool.blocks
            keep = Block[]
            for blk in blks
                # "Nothing is out on loan", which is the question, said directly.
                # It used to be asked of the free list — one entry spanning the
                # whole block — which is the same fact only while coalescing is
                # perfect, and makes the trim depend on the allocator's internals
                # rather than on its ledger.
                if isempty(blk.live)
                    rawfree(dev, blk.memory)
                else
                    push!(keep, blk)
                end
            end
            pool.blocks[kind] = keep
        end
    end
    return pool
end

"""
What a region is aligned to unless a tenant needs more.

Named because two places need the same number: this is `reserve!`'s default,
and the `Place` phase raises it to whatever its tenants ask for. Passing a
tenant's alignment straight through would LOWER it — a Metal storage buffer
wants 16 — and a region that used to start on a 256-byte boundary suddenly did
not, which cost 19% on the ray-tracing benchmark before it was caught.
"""
const REGION_ALIGN = 256

"""
    reserve!(pool, dev, kind, transients, bytes; align, blocksize) -> Region

The shared region for `kind`, grown to `bytes` if a tenant now needs more.

This is what `acquire!` is under: `acquire!` hands out a private slice, and two
plans that each take one can never share bytes however well either is placed.
`reserve!` hands every tenant of an arena the SAME slice, so the arena costs the
largest of them rather than their total.

**Which of the two you get is decided by what you are allocating, never by an
argument.** `Place` reserves, because a transient is scratch scoped to one run;
`allocate` — and so every `Buffer` and `Scalar` — acquires, because a persistent
resource holds data between runs and sharing its bytes would be silent
corruption. There is deliberately no flag here to get that wrong with.

Growing carves a fresh region, remaps the live tenants into it, and only then
releases the old one — in that order, so growth never tramples the bytes it is
copying tenants out of. Tenants keep their own offsets; nothing recompiles.

The caller registers itself with [`tenant!`](@ref) once it exists, because a plan
cannot be a tenant before it is a plan.
"""
function reserve!(pool::Pool, dev, kind, transients, bytes::Int;
                  align::Int = REGION_ALIGN, blocksize::Int = 64 << 20)
    lock(pool.lock) do
    a = arenaof(pool, kind)
    bytes = max(bytes, 1)
    req = constraintof(dev, kind, transients)
    # The REGION's absence is what says nothing has been placed here yet — not the
    # constraint's value, which a backend may legitimately leave as `nothing`.
    want = a.region === nothing ? req : mergeconstraints(dev, kind, a.constraint, req)
    # The fast path has to ask about the CONSTRAINT as well as the size: an
    # arena outlives the plan that sized it, and a block created for one plan's
    # usage bits may not permit what the next plan does with them. Skipping this
    # is memory that works until the one transient whose bit was missing is used.
    # `align` belongs in the fast path too, not just in `acquire!`. An arena
    # outlives the plan that sized it, and the region it kept was aligned to
    # whatever THAT plan asked for; handing it to a caller that needs more is a
    # region every one of whose tenant offsets is off its boundary. The
    # constraint check right below exists for the same reason and had the same
    # hole.
    if a.region !== nothing && bytes <= a.bytes && offset(a.region) % align == 0 &&
       compatible(dev, a.region.block.constraint, want)
        a.constraint = want
        return a.region
    end
    # Ask before allocating anything: a refusal must leave the arena exactly as it
    # was, not half-grown with a region nobody is placed in.
    for wr in tenants!(a)
        remappable(wr.value) || throw(ArgumentError(
            "a plan placed in arena $kind cannot be moved — it has been recorded, " *
            "and its command buffer holds the addresses the current region has. " *
            "Another plan now needs $(humanbytes(bytes)) there, which would grow the " *
            "arena and leave that recording pointing at freed storage. Build every " *
            "plan that shares a device before recording any of them."))
    end
    fresh = acquire!(pool, dev, kind, transients, max(bytes, a.bytes);
                     align, blocksize, constraint = want)
    old = a.region
    a.region, a.bytes, a.constraint = fresh, max(bytes, a.bytes), want
    for wr in tenants!(a)
        remap!(wr.value, kind, fresh)
    end
    # Retired, not released: the tenants were just copied OUT of these bytes by
    # a device copy, which has been recorded rather than run.
    old === nothing || retire!(pool, old)
    return fresh
    end
end

"""
    tenant!(pool, kind, x) -> x

Register `x` as holding the arena's shared bytes, so a later grow remaps it.

Idempotent: recompiling a plan re-registers the same object, and a list with it
twice would remap it twice and count it twice in `sharing`.
"""
function tenant!(pool::Pool, kind, x)
    lock(pool.lock) do
        a = arenaof(pool, kind)
        any(wr -> wr.value === x, tenants!(a)) || push!(a.tenants, WeakRef(x))
    end
    return x
end

"""
    untenant!(pool, kind, x)

Drop `x` from the arena, releasing the shared region once the last tenant goes.

The counterpart of [`tenant!`](@ref), called by `free!`. Refcounting by tenant
list rather than by a number: the list already has to be walked for `remap!`, and
a count that disagrees with it is a leak nobody can find.
"""
function untenant!(pool::Pool, kind, x)
    lock(pool.lock) do
    a = get(pool.arenas, kind, nothing)
    a === nothing && return nothing
    filter!(wr -> wr.value !== x && wr.value !== nothing, a.tenants)
    if isempty(a.tenants) && a.region !== nothing
        # Retired, not released. The last tenant leaving says nothing about
        # whether the device has finished running it — a plan freed right after
        # its final `run!` is the ordinary case, and its recording is still in
        # flight. This is why `free!(::Plan)` needs no precondition either.
        retire!(pool, a.region)
        a.region, a.bytes, a.constraint, a.lastrun = nothing, 0, nothing, nothing
    end
    return nothing
    end
end

"""
    takeover!(pool, kind, x) -> Bool

Record `x` as the tenant about to write this arena, and say whether it is taking
the bytes over from a DIFFERENT one.

What a backend asks before emitting the handover barrier. `sharing` is the wrong
question on its own: it says the arena has more than one tenant, which is true
for every run once two plans exist — so a plan run repeatedly, which is the
common case (playing one clip, running a recorded model), paid a full memory
barrier per frame to be ordered against itself. Its own hazards are its
schedule's business and already handled.

Recording and asking are one call because they must not drift: a backend that
asked without recording would emit forever, and one that recorded without asking
would emit never.
"""
function takeover!(p::Pool, kind, x)
    a = get(p.arenas, kind, nothing)
    a === nothing && return false
    prev = a.lastrun
    a.lastrun = x
    return prev !== nothing && prev !== x
end

"""
    sharing(pool, kind) -> Bool

Whether more than one live plan is placed in this arena.

What a backend asks before emitting the handover barrier: with one tenant there
is nothing to hand over from, and the barrier is pure cost.
"""
sharing(pool::Pool, kind) =
    (a = get(pool.arenas, kind, nothing); a === nothing ? false : length(tenants!(a)) > 1)
