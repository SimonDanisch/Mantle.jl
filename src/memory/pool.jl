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
    largestfree(pool, kind) -> Int

The biggest span any existing block of `kind` could still hand out.

What makes the capacity bound TIGHT rather than merely safe: a request this size
or smaller reaches no device allocation at all, so bounding it by what the device
has left would refuse a placement that costs nothing. Compatibility is not
consulted — this is a bound, and an over-estimate here can only be corrected by
`acquire!` going on to allocate, which the bound has already been checked against.
"""
largestfree(p::Pool, kind) =
    maximum((length(s) for b in blocksof(p, kind) for s in b.free); init = 0)

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
    compatible(dev, block_constraint, request) -> Bool

May a block created for `block_constraint` host a request needing `request`?

Core cannot answer this — "what memory is legal here" is exactly the backend's
semantics. It is why blocks are not interchangeable and why a pool cannot simply
be one large buffer.
"""
function compatible end

"""
One raw allocation, and which parts of it are unused.

`free` is kept sorted, disjoint and coalesced, so a released region merges with
its neighbours rather than fragmenting the block one plan at a time.
"""
mutable struct Block
    memory::Any
    bytes::Int
    kind::Any
    constraint::Any
    free::Vector{Span}
end
Block(memory, bytes::Int, kind, constraint) =
    Block(memory, bytes, kind, constraint, [Span(0, bytes)])

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
Blocks per arena kind, owned by a device.

Growth ADDS a block; it never reallocates an existing one. That is the whole
reason for a list: a plan holding offsets into a block must not have the ground
move under it because another plan needed more room.
"""
struct Pool
    blocks::Dict{Any,Vector{Block}}
end
Pool() = Pool(Dict{Any,Vector{Block}}())

blocksof(p::Pool, kind) = get!(() -> Block[], p.blocks, kind)

"""Every byte the pool has taken from the device, across all blocks."""
reserved(p::Pool) = sum(b -> b.bytes, Iterators.flatten(values(p.blocks)); init = 0)

"""First offset at or after `from` that satisfies `align`."""
alignup(from::Int, align::Int) = align <= 1 ? from : cld(from, align) * align

"""
Carve `bytes` out of `blk`, honouring `align`, or `nothing` if it does not fit.

First fit. Not best fit: the free list here is short (one entry per live plan per
block, not per transient — `Place` has already packed the transients), and best
fit costs a scan to save fragmentation that the coalescing release already
prevents in the common pattern of "the same plan recompiles at the same size".
"""
function carve!(blk::Block, bytes::Int, align::Int)
    for (i, s) in enumerate(blk.free)
        start = alignup(s.lower, align)
        start + bytes <= s.upper || continue
        deleteat!(blk.free, i)
        # Whatever alignment skipped, and whatever is left over, go back on the
        # list — dropping either leaks bytes that nothing can ever hand out.
        start > s.lower && insert!(blk.free, i, Span(s.lower, start))
        tail = Span(start + bytes, s.upper)
        isempty(tail) || insert!(blk.free, searchsortedfirst_lower(blk.free, tail.lower), tail)
        return Region(blk, Span(start, start + bytes))
    end
    return nothing
end

searchsortedfirst_lower(v::Vector{Span}, x::Int) =
    searchsortedfirst(v, x; by = s -> s isa Span ? s.lower : s)

"""
    acquire!(pool, dev, kind, transients, bytes; align) -> Region

`bytes` from a block that can host `transients`, growing the pool if none can.

A new block is at least `bytes` and at least `blocksize`, so a run of small plans
does not produce a device allocation each. `blocksize` is a kwarg rather than a
constant because the right value is a device property, not a Mantle opinion.
"""
function acquire!(pool::Pool, dev, kind, transients, bytes::Int;
                  align::Int = 256, blocksize::Int = 64 << 20)
    bytes = max(bytes, 1)
    want = constraintof(dev, kind, transients)
    blks = blocksof(pool, kind)
    for blk in blks
        compatible(dev, blk.constraint, want) || continue
        r = carve!(blk, bytes, align)
        r === nothing || return r
    end
    n = max(bytes, blocksize)
    blk = Block(rawalloc(dev, kind, n, want), n, kind, want)
    push!(blks, blk)
    r = carve!(blk, bytes, align)
    r === nothing && error("a fresh $n-byte block could not host $bytes bytes at " *
                           "alignment $align — the block size or the alignment is wrong")
    return r
end

"""
Give a region back, merging it with any neighbouring free space.

Coalescing here rather than at acquire time is what keeps a plan that recompiles
at the same size from walking up the block: release then acquire finds the span
it just gave back, whole.
"""
function release!(r::Region)
    blk = r.block
    i = searchsortedfirst_lower(blk.free, r.span.lower)
    # A double release is silent otherwise: the span goes on the list twice, two
    # later `acquire!`s hand out the same bytes, and the corruption surfaces
    # nowhere near here. Overlapping a neighbour is the only way that happens, so
    # checking two entries catches it at the moment it is committed.
    for j in (i - 1, i)
        1 <= j <= length(blk.free) && overlaps(blk.free[j], r.span) &&
            error("release!: $(r.span) overlaps free $(blk.free[j]) — released twice, " *
                  "or a Region outlived the block it came from")
    end
    insert!(blk.free, i, r.span)
    # merge with the previous, then the next; at most two joins per release
    if i > 1 && blk.free[i - 1].upper == blk.free[i].lower
        blk.free[i - 1] = Span(blk.free[i - 1].lower, blk.free[i].upper)
        deleteat!(blk.free, i)
        i -= 1
    end
    if i < length(blk.free) && blk.free[i].upper == blk.free[i + 1].lower
        blk.free[i] = Span(blk.free[i].lower, blk.free[i + 1].upper)
        deleteat!(blk.free, i + 1)
    end
    return nothing
end

"""
Hand every fully-unused block back to the device.

Explicit, and never from a finalizer. Lava already learned this one: its pool
free-list was pushed from the GC finalizer thread while another thread popped,
which gave a `ConcurrencyViolationError` and then a SIGSEGV. A pool that frees on
a finalizer is the same bug waiting for a different day.
"""
function trim!(pool::Pool, dev)
    for (kind, blks) in pool.blocks
        keep = Block[]
        for blk in blks
            if length(blk.free) == 1 && only(blk.free) == Span(0, blk.bytes)
                rawfree(dev, blk.memory)
            else
                push!(keep, blk)
            end
        end
        pool.blocks[kind] = keep
    end
    return pool
end
