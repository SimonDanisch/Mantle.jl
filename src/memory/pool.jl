# Backend-independent suballocation across plans: which block, which offset,
# when to grow and when to release. A backend supplies only `rawalloc`,
# `rawfree`, `constraintof` and `compatible`. `rawalloc` must be the device's
# rawest allocation (`vkAllocateMemory`), not a pooled wrapper, or the pool
# suballocates inside a second allocator it cannot see.

"""
    rawalloc(dev, kind, bytes, constraint) -> memory

One raw device allocation of at least `bytes`, legal for `constraint`. Called
once per block. Must not go through a backend pool, or the peak Mantle reports
describes its own bookkeeping rather than the device.
"""
function rawalloc end

"""Release a block's allocation. Called only when nothing references it."""
function rawfree end

"""
    maxalloc(dev) -> Int

The largest single allocation the device makes, usually far below
[`capacity`](@ref). Bounds a block, because a block is one allocation and
placement offsets index one contiguous range. The default `typemax(Int)` leaves
`capacity` as the only bound.
"""
maxalloc(dev) = typemax(Int)


"""
    constraintof(dev, kind, transients) -> constraint

What memory `transients` can legally share, in the form the backend's
[`compatible`](@ref) reads (a memory-type mask, usage flags). Derived from the
transients: on Vulkan an image's memory-type bits belong to the image, and a
block hosting several images must satisfy all of them.
"""
function constraintof end

"""
    mergeconstraints(dev, kind, a, b) -> constraint

One constraint satisfying both `a` and `b`, for an arena whose block must keep
serving every tenant placed in it. The backend defines the merge (Vulkan: buffer
usage bits union, image memory-type bits intersect); the default requires
`a == b`. Throws when no merge exists, because `nothing` is a valid constraint.
"""
mergeconstraints(dev, kind, a, b) = a == b ? a : throw(ArgumentError(
    "arena $kind cannot host this plan and the ones already placed in it: it " *
    "needs $b where they need $a, and one allocation cannot serve both. This is " *
    "not a size problem — a bigger arena would not help."))

"""
    compatible(dev, block_constraint, request) -> Bool

Whether a block created for `block_constraint` may host a request needing
`request`. Backend semantics, and the reason blocks are not interchangeable.
"""
function compatible end

"""One bin per power of two, which is as many as an `Int` length can reach."""
const NBINS = 8 * sizeof(Int)

"""
The unused spans of one block, indexed by start (`bylower`), by end (`byupper`)
and by size class (`bins[k]` holds the starts of spans with length in
`[2^(k-1), 2^k)`, `slot` their index there), so carving and coalescing are O(1)
amortised. A sorted span vector is about 10% faster below ~350 free spans but
O(free spans) per operation, with no bound once the pool serves general
allocation. Searching bins upward gives approximate best fit, which keeps long
spans whole.
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
The lowest bin whose spans are all at least `n` long. From here up any entry
fits unchecked; `binof(n)` straddles `n`, so [`carve!`](@ref) scans it.
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
    # Swap-with-last: a bin is unordered, so removal is O(1).
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
One raw allocation, its free spans, and `live`: offset => end of every region
[`carve!`](@ref) handed out and [`release!`](@ref) has not taken back. `live`
turns a double release into an error; coalescing leaves no trace of it in the
free list. `lock` is per block because `release!` gets only the region. Lock
order where both are held: pool, then block.
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
A slice of a block. A transient's offset from `Place` is relative to the region,
so `offset(region)` is added for `materialize!`.
"""
struct Region
    block::Block
    span::Span
end
offset(r::Region) = r.span.lower
Base.length(r::Region) = length(r.span)
memoryof(r::Region) = r.block.memory

"""
One arena's shared region and the plans placed in it. Every tenant starts at
offset 0, so the arena is sized to its largest tenant, not their sum. Safe
because every recording opens with a global barrier (emitted by the backend)
and a plan's transients are dead once another tenant runs. `tenants` is weak: a
plan nobody holds cannot run and needs no remap.
"""
mutable struct Arena
    region::Union{Nothing,Region}
    bytes::Int
    constraint::Any        # what every tenant placed here needs, merged
    tenants::Vector{WeakRef}
end
Arena() = Arena(nothing, 0, nothing, WeakRef[])

"""The live tenants as strong references, pruning collected ones. A `WeakRef` can
clear at any allocation, so callers hold these instead of rereading `.value`."""
function tenants!(a::Arena)
    live = Any[]
    filter!(a.tenants) do wr
        t = wr.value
        t === nothing && return false
        push!(live, t)
        return true
    end
    return live
end

"""
    remap!(tenant, kind, region)

Rebind a tenant's transients of arena `kind` to `region` after that arena grew.
The tenant's own offsets stay valid, so nothing recompiles. `kind` because a
tenant usually spans several arenas and only one moved.
"""
function remap! end

"""
    remappable(tenant) -> Bool

Whether `tenant` survives its arena moving; `true` by default. A recorded plan
does not: its commands hold the old region's addresses and a replay would
silently read freed storage. Growth asks every live tenant before moving.
"""
remappable(x) = true

# Lock-free stack a finalizer pushes onto with one CAS. A finalizer cannot wait
# on a contended `ReentrantLock` ("task switch not allowed from inside gc
# finalizer"), and `reclaim!(; wait = true)` holds the pool lock across a yield.
# `reclaim!` takes the whole stack on the owning thread.
mutable struct InboxNode
    item::Any
    next::Union{Nothing,InboxNode}
end

mutable struct Inbox
    @atomic head::Union{Nothing,InboxNode}
end
Inbox() = Inbox(nothing)

function Base.push!(ib::Inbox, item)
    node = InboxNode(item, nothing)
    old = @atomic ib.head
    while true
        node.next = old
        old, ok = @atomicreplace ib.head old => node
        ok && return ib
    end
end

# Everything pushed so far, newest first, leaving the inbox empty.
takeall!(ib::Inbox) = @atomicswap ib.head = nothing
Base.isempty(ib::Inbox) = (@atomic ib.head) === nothing

"""
    PoolPolicy()

When a pool trims and collects on its own. One per pool, changed as
`pool(dev).policy.trim_threshold = …`.

  * `trim_threshold`: reserved bytes above which [`autotrim!`](@ref) returns
    empty blocks. 1 GiB.
  * `trim_min_interval`: seconds between two such trims. 5.
  * `trim_full_gc_interval`: seconds between two full collections a trim may run
    when the incremental one found no empty block. 30.
  * `soft_cap`: reserved bytes above which an allocation collects before the
    pool grows ([`autocollect!`](@ref)); `0` disables it. 2 GiB.
  * `gc_mingap`, `gc_full_mingap`: seconds between two such collections, and
    between two full ones. 0.02 and 0.5.
  * `gc_budget`: share of wall time those collections may take. 0.05.

The other fields record when each last ran and what it cost. Trimming returns
memory to the device and drains the queue, so it is time-gated; collecting
before growth touches no driver, so it runs on the allocation path. At 2 GiB
SAM 2's encoder runs as fast as with a 3 GiB cap.
"""
mutable struct PoolPolicy
    trim_threshold::Int
    trim_min_interval::Float64
    trim_full_gc_interval::Float64
    soft_cap::Int
    gc_mingap::Float64
    gc_full_mingap::Float64
    gc_budget::Float64
    last_trim::Float64
    gc_last::Float64
    gc_full_last::Float64
    # Seconds the last collection took; the next gap is this over `gc_budget`.
    gc_lastcost::Float64
    gc_seconds::Float64
    gc_count::Int
    collecting::Threads.Atomic{Bool}
end
PoolPolicy() = PoolPolicy(1 << 30, 5.0, 30.0, 2 << 30, 0.02, 0.5, 0.05,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0, Threads.Atomic{Bool}(false))

"""
Blocks per arena kind, and the shared arena of each kind. Growth adds a block
and never reallocates one, so offsets into existing blocks stay valid. An arena
that outgrows its region gets a fresh carve and [`remap!`](@ref).
"""
struct Pool
    blocks::Dict{Any,Vector{Block}}
    arenas::Dict{Any,Arena}
    # Regions retired from a finalizer (`pending`), then stamped with a fence
    # (`retiring`, `retiring_at`); see `retire!` and `reclaim!`. Parallel
    # vectors with concrete `UInt64` fences so the per-run walk does not box;
    # a type parameter on `Pool` would cascade into the device types.
    pending::Inbox
    retiring::Vector{Region}
    retiring_at::Vector{UInt64}
    # Recorded plans whose argument memory holds device addresses, so a moving
    # resource patches them (`notify_move!`). Added by `record!`, removed by
    # `free!` and when the recording is dropped. Weak.
    movelisteners::Vector{WeakRef}
    # Bumped when a block is created or destroyed, so a cache derived from the
    # block list (Metal's recorded replay) checks staleness in one comparison.
    # Regions acquired or released inside existing blocks do not bump it.
    blockgen::Base.RefValue{Int}
    # Plans dropped without `free!`. Their teardown is not finalizer-safe (it
    # unlistens from `movelisteners` and destroys a command buffer), so the
    # finalizer pushes here and `reclaim!` frees them on the owning thread.
    pendingplans::Inbox
    lock::ReentrantLock
    # When this pool trims and collects by itself; see `PoolPolicy`.
    policy::PoolPolicy
end
Pool() = Pool(Dict{Any,Vector{Block}}(), Dict{Any,Arena}(),
              Inbox(), Region[], UInt64[], WeakRef[], Ref(0), Inbox(), ReentrantLock(),
              PoolPolicy())

"""
    retireplan!(pool, pl)

Take a dropped plan, from a finalizer. Pushes onto a lock-free stack and does
nothing else; [`reclaim!`](@ref) tears it down on the owning thread.
"""
retireplan!(p::Pool, pl) = (push!(p.pendingplans, pl); nothing)

arenaof(p::Pool, kind) = get!(Arena, p.arenas, kind)

"""
    movable(pool) -> Bool

Whether every live tenant of every arena survives its memory moving. `false`
once any plan holds a recording, whose commands name the current blocks; the
automatic and pressure trims refuse to run then.
"""
movable(p::Pool) =
    all(a -> all(remappable, tenants!(a)), values(p.arenas))

blocksof(p::Pool, kind) = get!(() -> Block[], p.blocks, kind)

"""Every byte the pool has taken from the device, across all blocks."""
reserved(p::Pool) = sum(b -> b.bytes, Iterators.flatten(values(p.blocks)); init = 0)

"""The bytes the pool has taken from the device for one kind of memory."""
reserved(p::Pool, kind) = sum(b -> b.bytes, get(p.blocks, kind, Block[]); init = 0)

"""
    largestfree(pool, kind) -> Int

The longest free span in any block of `kind`. A request this size or smaller
needs no device allocation. Ignores compatibility, so it is an upper bound.
"""
function largestfree(p::Pool, kind)
    best = 0
    for b in blocksof(p, kind)
        lock(b.lock) do
            # Only the highest non-empty bin can hold the longest span.
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

The largest single request of `kind` that can still be satisfied: the smaller
of [`maxalloc`](@ref) and the unreserved budget plus [`largestfree`](@ref).
"""
headroom(p::Pool, dev, kind) =
    min(maxalloc(dev), max(capacity(dev) - reserved(p), 0) + largestfree(p, kind))

"""
    arenaroom(pool, dev, kind) -> Int

The tallest placement arena `kind` can take: its current region, which
[`reserve!`](@ref) gives every tenant at offset 0, or [`headroom`](@ref),
whichever is larger. `headroom` alone counts the region as spent and would
refuse a plan no taller than an earlier tenant. An upper bound; `reserve!`
checks compatibility.
"""
arenaroom(p::Pool, dev, kind) = max(arenaof(p, kind).bytes, headroom(p, dev, kind))

"""First offset at or after `from` that satisfies `align`."""
alignup(from::Int, align::Int) = align <= 1 ? from : cld(from, align) * align

"""
How many entries of a straddling bin `carve!` checks before moving up a bin. A
miss costs at most a block a full scan would have avoided; an exhaustive scan
would be O(free spans) again.
"""
const BIN_SCAN = 8

"""
Carve `bytes` at `align` out of `blk`, or return `nothing`. Approximate best
fit: bins are searched upward from the request's own, keeping long spans whole.
"""
function carve!(blk::Block, bytes::Int, align::Int)
    lock(blk.lock) do
        fs = blk.free
        # Alignment costs at most `align - 1` bytes, so from this bin up any
        # entry fits.
        guaranteed = binceil(bytes + align - 1)
        for k in binof(bytes):NBINS
            v = fs.bins[k]
            isempty(v) && continue
            if k >= guaranteed
                return takespan!(blk, v[end], bytes, align)
            end
            # This bin straddles the request: check up to `BIN_SCAN` entries.
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
    # Return both the alignment gap and the remainder, or those bytes leak.
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
    acquire!(pool, dev, kind, transients, bytes; align = 256, blocksize = 64 MiB,
             constraint, maywait = true, onpressure) -> Region

A private region of `bytes` from a block that can host `transients`, growing the
pool if none can. A new block is at least `blocksize`, a device property the
caller passes. `constraint` replaces the one derived from `transients`.
`maywait = false` (inside an open recording) never waits for the device and
skips the pressure steps; `onpressure` replaces [`makeroom!`](@ref).
"""
function acquire!(pool::Pool, dev, kind, transients, bytes::Int;
                  align::Int = 256, blocksize::Int = 64 << 20, constraint = nothing,
                  maywait::Bool = true, onpressure = nothing)
    refuseretired(dev)
    bytes = max(bytes, 1)
    # A given `constraint` covers an arena's other tenants too; deriving it here
    # would size the block for this plan alone.
    want = constraint === nothing ? constraintof(dev, basekind(kind), transients) : constraint
    return underpressure(pool, dev, maywait, onpressure) do
        carveorgrow!(pool, dev, kind, bytes, align, blocksize, want, maywait)
    end
end

"""
    underpressure(f, pool, dev, maywait, onpressure = nothing) -> f()

Run `f()`. If the device refuses it for lack of memory and `maywait` is true,
run the pressure steps (`onpressure()`, default [`makeroom!`](@ref)) and `f()`
once more; a second refusal is final (docs/api.md, "An allocation beyond the
budget"). Must run outside the pool's lock: holding a `ReentrantLock` disables
finalizers, so a collection under it retires nothing.
"""
function underpressure(f, pool::Pool, dev, maywait::Bool, onpressure = nothing)
    try
        return f()
    catch err
        (maywait && isoutofmemory(err)) || rethrow()
    end
    onpressure === nothing ? makeroom!(pool, dev) : onpressure()
    return f()
end

"""
    carveorgrow!(pool, dev, kind, bytes, align, blocksize, want, maywait) -> Region

[`acquire!`](@ref)'s locked attempt: carve; reclaim and carve; wait, reclaim and
carve; then allocate a block. Out-of-memory propagates to [`underpressure`](@ref).
"""
function carveorgrow!(pool::Pool, dev, kind, bytes::Int, align::Int, blocksize::Int,
                      want, maywait::Bool)
    # Retired bytes may already be done on the device, so reclaiming before
    # growing keeps a deferred `free!` from doubling peak memory. `wait = true`
    # cannot block forever: `waitfor` declines unsubmitted work. The lock also
    # covers `blocksof`, since two allocating threads would `push!` the same
    # vector; it is reentrant, so `reclaim!` can take it again.
    lock(pool.lock) do
        r = carveany!(pool, dev, kind, bytes, align, want)
        r === nothing || return r
        if reclaim!(pool, dev) > 0
            r = carveany!(pool, dev, kind, bytes, align, want)
            r === nothing || return r
        end
        # `maywait = false` comes from `scratch!`, inside an open recording. A
        # waiting reclaim sweeps completed submissions, which pops the channel's
        # top hold frame (the open recording's), and its next `hold!` throws.
        if maywait && reclaim!(pool, dev; wait = true) > 0
            r = carveany!(pool, dev, kind, bytes, align, want)
            r === nothing || return r
        end
        blks = blocksof(pool, kind)
        limit = maxalloc(dev)
        bytes <= limit || throw(ArgumentError(
            "GPU memory request of $bytes bytes exceeds the device's single-allocation limit of $limit bytes"))
        n = min(max(bytes, blocksize), limit)
        blk = Block(rawalloc(dev, basekind(kind), n, want), n, kind, want)
        push!(blks, blk)
        pool.blockgen[] += 1
        r = carve!(blk, bytes, align)
        r === nothing && error("a fresh $n-byte block could not host $bytes bytes at " *
                               "alignment $align — the block size or the alignment is wrong")
        return r
    end
end

"""
    release!(r::Region)

Give a region back and coalesce it with its free neighbours, so a plan
recompiled at the same size gets its span back whole. Throws for a region that
is not live: released twice, or outlived its block.
"""
function release!(r::Region)
    blk = r.block
    lock(blk.lock) do
        lo, hi = r.span.lower, r.span.upper
        # Checked against `live`: a double release would hand the same bytes
        # out twice, and coalescing leaves nothing in the free list to check.
        held = get(blk.live, lo, -1)
        held == hi || error(
            "release!: $(r.span) was not handed out by this block — " *
            (held == -1 ?
             "nothing live starts at $lo, so this region was released already, or it " *
             "outlived the block it came from" :
             "the live region at $lo ends at $held, not $hi"))
        delete!(blk.live, lo)
        # Join the span ending at `lo` and the span starting at `hi`.
        fs = blk.free
        p = get(fs.byupper, lo, -1)
        p == -1 || (dropspan!(fs, p); lo = p)
        haskey(fs.bylower, hi) && (hi = dropspan!(fs, hi))
        addspan!(fs, lo, hi)
    end
    return nothing
end

"""
    fence(dev) -> UInt64

A monotone counter standing for everything submitted to `dev` so far (Vulkan: a
timeline-semaphore value, Metal: a retirement count). Core only hands it to
[`passed`](@ref) and [`waitfor`](@ref). `UInt64` so `reclaim!`'s per-run walk
does not allocate. A backend with nothing to count returns zero.
"""
function fence end

"""
    passed(dev, f) -> Bool

Whether `dev` has finished everything the [`fence`](@ref) value `f` stands for.
"""
function passed end

"""
    waitfor(dev, f) -> Bool

Block until [`passed`](@ref)`(dev, f)` and return whether it did. Returns
`false` without waiting when `f` names work not yet submitted: forcing a submit
would split an open recording, so the caller grows the pool instead. Never
submits or flushes.
"""
function waitfor end

"""
    retire!(pool, dev, r::Region)

Give a region back from a finalizer: push it onto a lock-free stack and nothing
else, since a finalizer must not take a lock or touch a free list another thread
is carving. [`reclaim!`](@ref) releases it on the owning thread once the device
is done. `dev` is passed because a `Pool` does not know its device, and a
backend method must order against that device's queue.
"""
retire!(p::Pool, dev, r::Region) = (push!(p.pending, r); nothing)

"""
    reclaim!(pool, dev; wait = false) -> Int

Release every retired region the device has finished with; return how many.
Two phases: the first call to see a region stamps it with [`fence`](@ref), and a
later call releases it once [`passed`](@ref), since work recorded before the
drop may still be in flight. Also frees plans dropped without `free!`. Cheap
and idempotent, so callable every frame. Backend arrays go back through their
channel instead (`retire!(ch, obj)`). `wait = true` also blocks on submitted
work via [`waitfor`](@ref); [`acquire!`](@ref) uses it before growing.
"""
function reclaim!(p::Pool, dev; wait::Bool = false)
    # Locked throughout: `acquire!` calls this from any allocating thread.
    # Explicit lock/try/unlock: a `lock() do` closure allocates 224 bytes per
    # call, and `run!` must not allocate.
    lock(p.lock)
    try
        # Plans first: tearing one down retires its regions into this pass.
        # `free!` is idempotent.
        node = takeall!(p.pendingplans)
        while node !== nothing
            free!(node.item)
            node = node.next
        end
        node = takeall!(p.pending)
        if node !== nothing
            f = fence(dev)
            while node !== nothing
                push!(p.retiring, node.item::Region)
                push!(p.retiring_at, f)
                node = node.next
            end
        end
        # `retiring_at` is sorted (`fence` is monotone, entries are appended),
        # so finished entries form a prefix and the first unfinished one ends
        # the loop: an idle frame asks `passed` once.
        freed = 0
        n = length(p.retiring)
        while freed < n
            f = p.retiring_at[freed + 1]
            (passed(dev, f) || (wait && waitfor(dev, f))) || break
            release!(p.retiring[freed + 1])
            freed += 1
        end
        if freed > 0
            deleteat!(p.retiring, 1:freed)
            deleteat!(p.retiring_at, 1:freed)
        end
        return freed
    finally
        unlock(p.lock)
    end
end

function trimready!(pool::Pool, dev)
    if supports_batch_queue(dev)
        ch = batchqueue(dev)
        holdsretired(ch) && drain!(ch)
    end
    reclaim!(pool, dev; wait = true)
    return trim!(pool, dev)
end

function withcollection(f, p::PoolPolicy)
    # No pool lock across GC: it disables the finalizers that retire regions.
    Threads.atomic_cas!(p.collecting, false, true) && return false
    try
        return f()
    finally
        p.collecting[] = false
    end
end

# After a pressure collection: return what it freed without another full one.
function trimcollected!(pool::Pool, dev)
    movable(pool) || return false
    return withcollection(pool.policy) do
        reclaimable(pool, dev) || return false
        supports_batch_queue(dev) && flush!(dev)
        trimready!(pool, dev)
        return true
    end
end

function collection!(f, p::PoolPolicy; continuing::Bool = false)
    t0 = time_ns()
    try
        return f()
    finally
        p.gc_last = time()
        elapsed = (time_ns() - t0) / 1e9
        p.gc_lastcost = elapsed + (continuing ? p.gc_lastcost : 0.0)
        p.gc_seconds += elapsed
        continuing || (p.gc_count += 1)
    end
end

function quiesce_pool!(flush, drain, pool::Pool, dev)
    movable(pool) || return false
    p = pool.policy
    return withcollection(p) do
        # Submission holds must be released before GC can find dropped owners.
        # The wait is not charged to the GC budget.
        flush()
        collection!(p) do
            GC.gc(true)
            p.gc_full_last = time()
        end
        drain()
        reclaim!(pool, dev; wait = true)
        return true
    end
end

"""
    makeroom!(pool, dev) -> pool

Flush to release submission holds, run a full collection, reclaim retired
objects and regions, and trim empty blocks. What a placement that does not fit
tries before giving up; refused while a recording pins the pool
([`movable`](@ref)). The flush matters: an upload's staging region returns only
when the channel sweeps its one-shot, and can keep a whole block alive.
"""
function makeroom!(pool::Pool, dev)
    return makeroom!(() -> quiesce_pool!(pool, dev), pool, dev)
end

function makeroom!(quiesce, pool::Pool, dev)
    quiesce() && trim!(pool, dev)
    return pool
end

function quiesce_pool!(pool::Pool, dev)
    flush = () -> (supports_batch_queue(dev) && flush!(dev))
    drain = () -> if supports_batch_queue(dev)
        ch = batchqueue(dev)
        holdsretired(ch) && drain!(ch)
    end
    return quiesce_pool!(flush, drain, pool, dev)
end

"""
    autotrim!(pool, dev) -> Bool

[`makeroom!`](@ref) run on an allocation: only above `policy.trim_threshold`,
at most every `policy.trim_min_interval` seconds, only when something can come
back, and never while a plan holds a recording ([`movable`](@ref)). Returns
whether it trimmed. Regions return to their block through finalizers, and an
incremental collection misses regions that survived a few frames, so when it
finds nothing a full collection runs, at most every
`policy.trim_full_gc_interval` seconds because it costs tens of milliseconds.
"""
function autotrim!(pool::Pool, dev)
    p = pool.policy
    reserved(pool) < p.trim_threshold && return false
    return withcollection(p) do
        now = time()
        now - p.last_trim < p.trim_min_interval && return false
        p.last_trim = now
        full = collection!(p) do
            GC.gc(false)
            return !reclaimable(pool, dev)
        end
        full && now - p.gc_full_last < p.trim_full_gc_interval && return false
        movable(pool) || return false
        supports_batch_queue(dev) && flush!(dev)
        if full
            collection!(p; continuing = true) do
                GC.gc(true)
                p.gc_full_last = time()
            end
            reclaimable(pool, dev) || return false
        end
        trimready!(pool, dev)
        return true
    end
end

"""
    reclaimable(pool, dev) -> Bool

Whether the automatic trim can return anything without waiting for the device:
an empty block after a non-waiting [`reclaim!`](@ref), or destroys the device's
channel holds. Does not see buffers held by an unsubmitted recording: that
needs a flush, which would stall a render loop every trim interval.
"""
function reclaimable(pool::Pool, dev)
    reclaim!(pool, dev)
    lock(pool.lock) do
        any(b -> isempty(b.live), Iterators.flatten(values(pool.blocks)))
    end && return true
    return supports_batch_queue(dev) && holdsretired(batchqueue(dev))
end

"""
    autocollect!(drain, pool; pressure = nothing, blocking = false) -> Bool

A collection on the allocation path, for a pool past `policy.soft_cap` that is
about to grow or a device heap under `pressure`, followed by `drain()` to return
what it freed. Returns whether it ran. Spaced by the larger of
`policy.gc_mingap` and the last collection's cost over `policy.gc_budget`; a
full collection follows at most every `policy.gc_full_mingap`.

`pressure` is the fraction of the device heap in use: below 0.75 (0.5 when
`blocking`) nothing runs, and `blocking`, `> 0.9` and `> 0.95` each double the
budget. Both triggers share one history so an allocation cannot collect twice.
A pressure collection goes full only when the pool also exceeds the soft cap.
"""
function autocollect!(drain, pool::Pool; pressure = nothing, blocking::Bool = false)
    pressure !== nothing && pressure < (blocking ? 0.5 : 0.75) && return false
    p = pool.policy
    return withcollection(p) do
        now = time()
        budget = p.gc_budget
        if pressure !== nothing
            blocking && (budget *= 2)
            pressure > 0.9 && (budget *= 2)
            pressure > 0.95 && (budget *= 2)
        end
        now - p.gc_last < max(p.gc_mingap, p.gc_lastcost / budget) && return false
        collection!(p) do
            GC.gc(false)
            drain()
            full = pressure === nothing || (p.soft_cap > 0 && reserved(pool) >= p.soft_cap)
            if full && now - p.gc_full_last >= p.gc_full_mingap
                GC.gc(true)
                drain()
                p.gc_full_last = time()
            end
        end
        return true
    end
end

"""
    trim!(pool, dev) -> pool

Return every block with no live region to the device. Never from a finalizer:
a GC-thread free racing an allocating thread corrupts the block list (Lava's
pool hit a `ConcurrencyViolationError`, then a SIGSEGV).
"""
function trim!(pool::Pool, dev)
    lock(pool.lock) do
        for (kind, blks) in pool.blocks
            keep = Block[]
            for blk in blks
                # Ask `live`, not the free list: they agree only while
                # coalescing is perfect.
                if isempty(blk.live)
                    rawfree(dev, blk.memory)
                else
                    push!(keep, blk)
                end
            end
            length(keep) == length(blks) || (pool.blockgen[] += 1)
            pool.blocks[kind] = keep
        end
    end
    return pool
end

"""
Default region alignment. `Place` raises it to what its tenants need and never
lowers it (a Metal storage buffer asks for 16); a region off a 256-byte
boundary cost 19% on the ray-tracing benchmark.
"""
const REGION_ALIGN = 256

"""
    reserve!(pool, dev, kind, transients, bytes; align, blocksize) -> Region

The shared region of arena `kind`, grown to `bytes` when a tenant needs more.
Every tenant gets the same region, so the arena costs its largest tenant.
`Place` reserves (transients are per-run scratch); persistent resources use
[`acquire!`](@ref), since shared bytes would corrupt data kept between runs.

Growth carves a fresh region, remaps the tenants into it, then retires the old
one; tenants keep their offsets. The caller registers with [`tenant!`](@ref)
once it exists.
"""
function reserve!(pool::Pool, dev, kind, transients, bytes::Int;
                  align::Int = REGION_ALIGN, blocksize::Int = 64 << 20)
    refuseretired(dev)
    # Pressure steps wrap the locked part (see `underpressure`), so growth
    # below calls `carveorgrow!`, not `acquire!`.
    underpressure(pool, dev, true) do
    lock(pool.lock) do
    a = arenaof(pool, kind)
    bytes = max(bytes, 1)
    req = constraintof(dev, basekind(kind), transients)
    # No region, not a `nothing` constraint, means no tenant yet: `nothing` is a
    # valid backend constraint.
    want = a.region === nothing ? req : mergeconstraints(dev, basekind(kind), a.constraint, req)
    # The fast path checks alignment and constraint too: the region was sized,
    # aligned and typed for an earlier tenant.
    if a.region !== nothing && bytes <= a.bytes && offset(a.region) % align == 0 &&
       compatible(dev, a.region.block.constraint, want)
        a.constraint = want
        return a.region
    end
    # Recorded tenants hold the region's addresses. Buffers: `notify_move!`
    # below patches the plan's BDAs. Images: the commands name image handles,
    # so the recording is invalidated and the next `run!` records again. Any
    # other tenant cannot be patched and refuses; the arena stays unchanged.
    live = tenants!(a)
    for t in live
        remappable(t) && continue
        if t isa Plan
            basekind(kind) isa Images && invalidate!(t)
        else
            throw(ArgumentError(
                "a tenant of arena $kind cannot be moved — it has been recorded, " *
                "and its commands hold the addresses the current region has. " *
                "Another plan now needs $(humanbytes(bytes)) there, which would grow the " *
                "arena and leave that recording pointing at freed storage."))
        end
    end
    fresh = carveorgrow!(pool, dev, kind, max(bytes, a.bytes), align, blocksize, want, true)
    old = a.region
    a.region, a.bytes, a.constraint = fresh, max(bytes, a.bytes), want
    for t in live
        remap!(t, kind, fresh)
    end
    # Retired, not released: the remap's copies out of these bytes are recorded,
    # not yet executed.
    old === nothing || retire!(pool, dev, old)
    # Recorded addresses into the old region shift by one delta. Images were
    # invalidated above; `notify_move!` invalidates plans it cannot patch.
    old === nothing || basekind(kind) isa Images ||
        notify_move!(pool, deviceaddress(dev, old), deviceaddress(dev, fresh), length(old))
    return fresh
    end
    end
end

"""
    tenant!(pool, kind, x) -> x

Register `x` as a tenant of arena `kind` so a later growth remaps it.
Idempotent, since recompiling a plan registers it again.
"""
function tenant!(pool::Pool, kind, x)
    lock(pool.lock) do
        a = arenaof(pool, kind)
        any(t -> t === x, tenants!(a)) || push!(a.tenants, WeakRef(x))
    end
    return x
end

"""
    untenant!(pool, dev, kind, x)

Drop `x` from arena `kind`, retiring the region once the last tenant leaves.
Called by `free!`. The tenant list is the refcount, so the two cannot disagree.
"""
function untenant!(pool::Pool, dev, kind, x)
    lock(pool.lock) do
    a = get(pool.arenas, kind, nothing)
    a === nothing && return nothing
    filter!(wr -> wr.value !== x && wr.value !== nothing, a.tenants)
    if isempty(a.tenants) && a.region !== nothing
        # Retired, not released: the last tenant's final `run!` may still be in
        # flight, so `free!(::Plan)` needs no precondition.
        retire!(pool, dev, a.region)
        a.region, a.bytes, a.constraint = nothing, 0, nothing
    end
    return nothing
    end
end

"""
    sharing(pool, kind) -> Bool

Whether more than one live plan is placed in arena `kind`. Informational only:
every recording opens with a global barrier regardless.
"""
sharing(pool::Pool, kind) =
    (a = get(pool.arenas, kind, nothing); a === nothing ? false : length(tenants!(a)) > 1)

# ── Moves ────────────────────────────────────────────────────────────────────
#
# A moved resource (`resize!`, a grown arena) changes device addresses that
# recorded plans packed into argument memory or recording-owned scratch. The
# recording is never rewritten: the next run's submission writes the new
# addresses ahead of it, like a `GPURef` update. Each plan's table of
# (region, offset) per address is built at `record!`, so a move is a lookup.

"""Register `pl` for move notification. Called by `record!`; idempotent."""
function listen_moves!(pool::Pool, pl)
    lock(pool.lock) do
        filter!(wr -> wr.value !== nothing, pool.movelisteners)
        any(wr -> wr.value === pl, pool.movelisteners) ||
            push!(pool.movelisteners, WeakRef(pl))
    end
    return nothing
end

"""Drop `pl` from move notification. Called by `free!` and when its recording is dropped."""
function unlisten_moves!(pool::Pool, pl)
    lock(pool.lock) do
        filter!(wr -> wr.value !== nothing && wr.value !== pl, pool.movelisteners)
    end
    return nothing
end

"""
    deviceaddress(dev, memory) -> UInt64

Where a backend allocation starts in the GPU address space (Vulkan
`memory.address`, Metal `buf.gpuAddress`). The only fact about a move core
needs from a backend.
"""
function deviceaddress end

deviceaddress(dev, r::Region) = deviceaddress(dev, memoryof(r)) + UInt64(offset(r))

"""
    notify_move!(pool, old_base, new_base, nbytes)

`[old_base, old_base + nbytes)` moved to `new_base`. For each recorded plan
holding an address in that range, queue a patch of every (region, offset) that
holds it, and re-key the table so the next move still finds it. Plans on a
backend without patching ([`patchable`](@ref)) are invalidated instead. Called
where the move happens (`resize!`, `reserve!`), never at run time.
"""
function notify_move!(pool::Pool, old_base::UInt64, new_base::UInt64, nbytes::Int)
    old_base == new_base && return nothing
    # Invalidated after the walk: `invalidate!` reaches `unlisten_moves!`,
    # whose `filter!` on this vector would skip entries mid-iteration.
    drop = Any[]
    lock(pool.lock) do
        filter!(wr -> wr.value !== nothing, pool.movelisteners)
        for wr in pool.movelisteners
            pl = wr.value
            # A `WeakRef` can clear at any safepoint and this loop allocates, so
            # the `filter!` above does not guarantee `pl` is alive.
            pl === nothing && continue
            # A recording that cannot be rewritten is dropped: it has no patch
            # table and would keep naming the old storage.
            if !patchable(pl.graph.dev)
                push!(drop, pl)
                continue
            end
            tab = pl.patchtab
            isempty(tab) && continue
            moved = [a for a in keys(tab) if old_base <= a < old_base + nbytes]
            for a in moved
                offs = pop!(tab, a)
                na = a - old_base + new_base
                for (r, off) in offs
                    push!(pl.pending_patches, (r, off, na))
                end
                tab[na] = offs
            end
        end
        nothing
    end
    for pl in drop
        invalidate!(pl)
    end
    return nothing
end
