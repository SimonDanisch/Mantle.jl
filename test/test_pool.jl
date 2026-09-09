# The suballocator, exercised with no backend whatsoever.
#
# That this file needs neither Lava nor KernelAbstractions is the assertion: if
# `Pool` ever needs a real device to be tested, some policy has leaked into a
# backend and a second backend will have to copy it.

using Test, Random
import Mantle
const M = Mantle

"""
A device that allocates nothing and counts what it was asked for.

`allocs` is the whole point of the fixture — the headline property is "a second
acquire does not reach the device", and that is only observable by counting.
"""
# `<: M.Device`, because that is what it stands in for. It was a bare struct
# while every verb it reached took `dev` untyped; `Buffer` now normalises its
# device argument with `todevice`, which accepts a `Device` and a KA backend and
# nothing else — deliberately, so a mistake fails where it is made. A test double
# that plays a device has to be one.
struct FakeDev <: M.Device
    allocs::Vector{Int}
end
FakeDev() = FakeDev(Int[])

M.rawalloc(d::FakeDev, kind, bytes, c) = (push!(d.allocs, bytes); zeros(UInt8, bytes))
M.rawfree(::FakeDev, mem) = nothing
M.constraintof(::FakeDev, kind, ts) = kind          # kind IS the constraint here
M.compatible(::FakeDev, a, b) = a === b
# `nothing` = "this element type asks for nothing beyond the ordinary", which
# sends `acquire!` back to `constraintof` above — the behaviour these tests were
# written against. Needed because `persistentarray` now ASKS every device this;
# it always should have, and until it did, `bufferusage` was a hook that both
# real backends implemented and nothing ever called.
M.bufferusage(::FakeDev, ::Type) = nothing

@testset "Pool: a second acquire does not reach the device" begin
    d, p = FakeDev(), M.Pool()
    r1 = M.acquire!(p, d, :buf, nothing, 1000; blocksize = 4096)
    @test d.allocs == [4096]                        # one block, rounded to blocksize
    r2 = M.acquire!(p, d, :buf, nothing, 1000; blocksize = 4096)
    @test d.allocs == [4096]                        # THE POINT
    @test M.offset(r1) != M.offset(r2)
    @test length(r1) == length(r2) == 1000
end

@testset "Pool: release coalesces" begin
    d, p = FakeDev(), M.Pool()
    a = M.acquire!(p, d, :buf, nothing, 1000; blocksize = 4096)
    b = M.acquire!(p, d, :buf, nothing, 1000; blocksize = 4096)
    M.release!(a); M.release!(b)
    # Only correct if the two freed spans merged: neither alone has room for 2048
    # at offset 0, and a plan that recompiles at the same size must find the span
    # it gave back whole rather than walking up the block.
    c = M.acquire!(p, d, :buf, nothing, 2048; blocksize = 4096)
    @test d.allocs == [4096]
    @test M.offset(c) == 0
end

@testset "Pool: growth adds a block and does not move the old one" begin
    d, p = FakeDev(), M.Pool()
    held = M.acquire!(p, d, :buf, nothing, 3000; blocksize = 4096)
    where_it_was = M.offset(held)
    M.acquire!(p, d, :buf, nothing, 4096; blocksize = 4096)     # cannot fit; must grow
    @test length(d.allocs) == 2
    # The reason the pool is a LIST of blocks: a plan holding offsets must not
    # have the ground move because another plan needed more room.
    @test M.offset(held) == where_it_was
end

@testset "Pool: incompatible kinds never share a block" begin
    d, p = FakeDev(), M.Pool()
    M.acquire!(p, d, :buf, nothing, 8; blocksize = 512)
    M.acquire!(p, d, :img, nothing, 8; blocksize = 512)
    @test length(d.allocs) == 2
    @test M.reserved(p) == 1024
end

@testset "Pool: alignment, and the skipped bytes are not leaked" begin
    d, p = FakeDev(), M.Pool()
    M.acquire!(p, d, :buf, nothing, 1; blocksize = 8192)         # sits at 0
    r = M.acquire!(p, d, :buf, nothing, 1; align = 1024, blocksize = 8192)
    @test M.offset(r) % 1024 == 0
    @test M.offset(r) > 0
    # The gap alignment skipped went back on the free list rather than vanishing:
    # something small must still fit in front of `r`.
    small = M.acquire!(p, d, :buf, nothing, 1; blocksize = 8192)
    @test M.offset(small) < M.offset(r)
    @test d.allocs == [8192]
end

@testset "Pool: trim! frees only fully-unused blocks" begin
    d, p = FakeDev(), M.Pool()
    a = M.acquire!(p, d, :buf, nothing, 8; blocksize = 512)
    M.acquire!(p, d, :img, nothing, 8; blocksize = 512)
    M.release!(a)
    M.trim!(p, d)
    # `:buf`'s block is now entirely free and goes back; `:img`'s is still held.
    @test M.reserved(p) == 512
end

# ── DeviceArray: a typed handle that owns nothing ─────────────────────────────

@testset "DeviceArray: typed view on a region, owning nothing" begin
    d, p = FakeDev(), M.Pool()
    a = M.allocate(p, d, :buf, Float32, 16, 4; blocksize = 4096)
    @test eltype(a) == Float32
    @test size(a) == (16, 4) && length(a) == 64 && sizeof(a) == 256
    @test M.offset(a) == M.offset(M.region(a))

    # Dropping the handle frees NOTHING. No finalizer, nothing for the GC thread
    # to race — which is the failure this codebase has paid for three times.
    a = nothing; GC.gc()
    @test M.reserved(p) == 4096
    b = M.allocate(p, d, :buf, Float32, 16, 4; blocksize = 4096)
    @test M.offset(b) != 0                       # the first region is still held

    # Freeing is explicit, and only through the region.
    M.release!(M.region(b))
    c = M.allocate(p, d, :buf, Float32, 16, 4; blocksize = 4096)
    @test M.offset(c) == M.offset(b)             # reused exactly
    @test d.allocs == [4096]
end

@testset "DeviceArray: a region too small is an error, not a silent overrun" begin
    d, p = FakeDev(), M.Pool()
    r = M.acquire!(p, d, :buf, nothing, 16; blocksize = 4096)
    @test_throws ArgumentError M.DeviceArray{Float32}(r, (16,))   # needs 64, has 16
    @test M.DeviceArray{Float32}(r, (4,)) isa M.DeviceArray{Float32,1}
end

# ── persistent resources: one implementation, no backend ──────────────────────

M.rawalloc(d::FakeDev, ::M.Persistent, bytes, c) = (push!(d.allocs, bytes); zeros(UInt8, bytes))
M.blocksize(::FakeDev) = 1 << 16
# A slab is host memory, so the three transfer verbs are core's over one answer.
# They used to be written out here with `reinterpret`, which is a fourth copy of
# what the host and Metal backends each had — and the broken spelling of it:
# `reinterpret` refuses any element type with padding, so this fixture could
# only ever move `Float32`. The testset below is the one that would have caught
# that, and it is here rather than beside a backend because it needs no device.
M.hostspan(::FakeDev, slab::Vector{UInt8}) = (pointer(slab), length(slab))
M.awaitwrites(::FakeDev) = nothing        # nothing is queued, so nothing to wait for
M.upload!(d::FakeDev, a::M.DeviceArray, first, data) = M.hostupload!(d, a, first, data)
M.download(d::FakeDev, a::M.DeviceArray) = M.hostdownload(d, a)
M.devicecopy!(d::FakeDev, dst, src, n) = M.hostdevicecopy!(d, dst, src, n)
M.deviceview(d::FakeDev, a) = a

# One transfer path, and the element type it used to fail on.
#
# `hostview`, `upload!`, `download` and `devicecopy!` lived three times: in the
# host backend as `wrapbytes`, in the Metal backend as its own `hostview`, and
# here as a `reinterpret`. The Metal copy's own comment named the reason they
# had to agree — `reinterpret` throws "Padding of type X is not compatible with
# type UInt8", and Hikari's `LightBVHNode` is 60 bytes holding 54 of fields, so
# every scene with a light BVH failed on it. They are one function over
# `hostspan` now, and this pins the property that made merging them safe.
#
# No device: a `FakeDev` is a slab and two one-line answers, which is the whole
# interface a host-addressable backend implements.
struct Padded
    a::Int32
    b::Int8
end

@testset "the shared transfer path moves a padded struct" begin
    d = FakeDev(); p = M.Pool()
    @eval M.pool(::$(typeof(d))) = $p
    vals = [Padded(Int32(i), Int8(2i)) for i in 1:8]

    b = M.Buffer(d, vals)
    @test M.download(d, b.store) == vals            # reinterpret threw here

    M.upload!(d, b.store, 3, [Padded(Int32(99), Int8(7))])
    got = M.download(d, b.store)
    @test got[3] == Padded(Int32(99), Int8(7))
    @test got[[1, 2, 4, 5, 6, 7, 8]] == vals[[1, 2, 4, 5, 6, 7, 8]]

    dst = M.Buffer(d, fill(Padded(Int32(0), Int8(0)), 8))
    M.devicecopy!(d, dst.store, b.store, 8)
    @test M.download(d, dst.store) == got

    # The bounds check is why this is one function and not an `unsafe_wrap` per
    # call site: a region the placer overlapped is caught, not silently read.
    @test_throws ArgumentError M.hostview(Int64, pointer(zeros(UInt8, 8)), 8, 0, (100,))
    @test_throws ArgumentError M.hostview(Int64, pointer(zeros(UInt8, 64)), 64, 8, (8,))
end

@testset "Buffer/GPURef are core types over pool regions" begin
    d = FakeDev(); p = M.Pool(); dev = d
    # the pool has to be reachable from the device, as it is for a real one
    @eval M.pool(::$(typeof(d))) = $p

    b = M.Buffer(dev, Float32[1, 2, 3]; capacity = 8)
    @test length(b) == 3 && M.capacity(b) == 8 && eltype(b) == Float32
    @test Array(b) == Float32[1, 2, 3]
    @test length(d.allocs) == 1                     # ONE block, not one per resource

    s = M.GPURef(dev, 7.0f0)
    @test length(d.allocs) == 1                     # …shared with the buffer
    M.update!(s, 9.0f0)

    # within capacity: no reallocation, no new region
    M.update!(b, 4:6, Float32[4, 5, 6])
    @test length(b) == 6 && Array(b) == Float32[1, 2, 3, 4, 5, 6]
    @test length(d.allocs) == 1

    # past capacity: a fresh region, the contents carried over, the old one back
    before = M.offset(M.region(b.store))
    resize!(b, 32)
    @test M.capacity(b) == 32
    @test Array(b) == Float32[1, 2, 3, 4, 5, 6]     # devicecopy! ran before release!
    @test M.offset(M.region(b.store)) != before
end

@testset "release! catches a double release instead of corrupting" begin
    d, p = FakeDev(), M.Pool()
    a = M.acquire!(p, d, :buf, nothing, 64; blocksize = 4096)
    b = M.acquire!(p, d, :buf, nothing, 64; blocksize = 4096)
    M.release!(a)
    # Without the guard this puts a's span back twice, and the NEXT two acquires
    # hand out the same bytes — corruption with no error anywhere near it.
    @test_throws ErrorException M.release!(a)
    # …and a legitimate release of a different region still works.
    M.release!(b)
    c = M.acquire!(p, d, :buf, nothing, 128; blocksize = 4096)
    @test M.offset(c) == 0                      # both spans coalesced back
    @test d.allocs == [4096]
end

@testset "release! catches a double release the free list cannot see" begin
    # The case the old neighbour check could not reach and the reason the block
    # keeps a `live` ledger. Release `a`, then release its neighbour `b`: `b`
    # coalesces with `a`, so no free span starts where `b` did any more. A second
    # release of `b` finds nothing that looks like itself, and on the free list
    # alone it reads as a fresh region being given back.
    d, p = FakeDev(), M.Pool()
    a = M.acquire!(p, d, :buf, nothing, 64; blocksize = 4096)
    b = M.acquire!(p, d, :buf, nothing, 64; blocksize = 4096)
    c = M.acquire!(p, d, :buf, nothing, 64; blocksize = 4096)
    M.release!(a)
    M.release!(b)
    @test_throws ErrorException M.release!(b)
    M.release!(c)
end

# ── The rewrite, checked against its own invariants ───────────────────────────
#
# `Block`'s free list stopped being a sorted `Vector{Span}` and became size-class
# bins plus two adjacency dictionaries, so that carving and releasing are O(1)
# rather than O(number of live regions in the block). The tests above pin the
# behaviours that were already understood; this one exists because a suballocator
# is exactly the kind of code whose bugs are silent — two live regions on the
# same bytes produces no error anywhere, only wrong data somewhere else later.
#
# So it is a randomised sequence checked against the properties that must hold no
# matter what the internals do, rather than against a second implementation:
# every live region disjoint, in range and aligned; every byte accounted for; and
# a pool that has given everything back holding exactly one free span per block.

"""Every free span in `blk`, recovered from the bins, sorted."""
freespans(blk) = sort([M.Span(lo, hi) for (lo, hi) in blk.free.bylower]; by = s -> s.lower)

"""
Are `blk`'s live regions and free spans a partition of it?

The single strongest statement about an allocator: disjoint, in range, and
covering every byte. Overlap is the corruption, and a gap is a leak.
"""
function partitions(blk)
    pieces = vcat(freespans(blk),
                  [M.Span(lo, hi) for (lo, hi) in blk.live])
    sort!(pieces; by = s -> s.lower)
    at = 0
    for s in pieces
        s.lower == at || return false          # a gap, or an overlap
        at = s.upper
    end
    return at == blk.bytes
end

@testset "Pool: randomised acquire/release keeps the block partitioned" begin
    for seed in 1:40
        rng = Random.Xoshiro(seed)
        d, p = FakeDev(), M.Pool()
        live = Any[]
        for _ in 1:400
            if !isempty(live) && rand(rng) < 0.45
                M.release!(popat!(live, rand(rng, 1:length(live))))
            else
                n = rand(rng, 1:3000)
                a = rand(rng, (1, 8, 64, 256, 1024))
                push!(live, M.acquire!(p, d, :buf, nothing, n; align = a, blocksize = 1 << 16))
            end
        end

        blocks = M.blocksof(p, :buf)
        @test all(partitions, blocks)

        # Alignment and length are what the caller was promised, and they have to
        # survive every split and merge the sequence above performed.
        @test all(r -> M.offset(r) + length(r) <= r.block.bytes, live)
        @test all(r -> r.block.live[M.offset(r)] == M.offset(r) + length(r), live)

        # Regions are pairwise disjoint WITHIN a block — implied by the partition
        # above, but asserted directly so a failure says which property broke.
        byblock = Dict{Any, Vector{Any}}()
        for r in live
            push!(get!(Vector{Any}, byblock, r.block), r)
        end
        for (_, rs) in byblock
            sort!(rs; by = M.offset)
            @test all(i -> M.offset(rs[i]) + length(rs[i]) <= M.offset(rs[i + 1]),
                      1:(length(rs) - 1))
        end

        # Give everything back: each block collapses to one free span, and every
        # block is then trimmable. A leaked byte shows up here as a block that
        # will not go.
        foreach(M.release!, live)
        @test all(b -> freespans(b) == [M.Span(0, b.bytes)], blocks)
        @test all(b -> isempty(b.live), blocks)
        M.trim!(p, d)
        @test M.reserved(p) == 0
    end
end

@testset "Pool: alignment is honoured under fragmentation" begin
    # The bounded per-bin scan can decline a span it could have used, which costs
    # a block and never a wrong offset. This is the assertion that separates the
    # two: whatever it hands back is aligned.
    rng = Random.Xoshiro(7)
    d, p = FakeDev(), M.Pool()
    live = Any[]
    for _ in 1:600
        if !isempty(live) && rand(rng) < 0.4
            M.release!(popat!(live, rand(rng, 1:length(live))))
        else
            a = rand(rng, (256, 512, 4096))
            r = M.acquire!(p, d, :buf, nothing, rand(rng, 1:5000); align = a,
                           blocksize = 1 << 16)
            @test M.offset(r) % a == 0
            push!(live, r)
        end
    end
end

@testset "Pool: bins" begin
    # `binof` is the bin a length lands in; `binceil` is the first bin every one
    # of whose members is long enough. `carve!` takes an entry unseen from
    # `binceil` upward, so the two disagreeing by one is a span handed out that
    # is too short — silent, and the worst failure this file can have.
    @test M.binof(1) == 1 && M.binof(2) == 2 && M.binof(3) == 2 && M.binof(4) == 3
    for n in 1:2048
        k = M.binceil(n)
        @test k == 1 || (1 << (k - 2)) < n          # not needlessly high
        @test (1 << (k - 1)) >= n                   # every span in bin k fits n
        @test M.binof(n) <= k
        # and the bin a length reports really does contain it
        @test (1 << (M.binof(n) - 1)) <= n < (1 << M.binof(n))
    end
end

@testset "the sharing invariant: transients share, persistents never do" begin
    # The property that makes a device-owned arena safe to use rather than a
    # footgun. `reserve!` hands every tenant the SAME bytes because a transient
    # is scratch scoped to one run; `acquire!` — and so `allocate`, and so every
    # `Buffer` and `GPURef` — hands out a private slice because a persistent
    # resource holds data between runs. Which one you get is decided by WHAT you
    # allocate, never by an argument, so there is no flag to get it wrong with.
    #
    # Asserted rather than documented: the failure mode of getting this backwards
    # is two live buffers silently on the same bytes, which no test of either one
    # would notice.
    d, p = FakeDev(), M.Pool()

    a = M.reserve!(p, d, :buf, nothing, 1024; blocksize = 4096)
    b = M.reserve!(p, d, :buf, nothing, 512; blocksize = 4096)
    @test M.offset(a) == M.offset(b)            # same arena, same bytes
    @test p.arenas[:buf].bytes == 1024          # sized to the LARGEST, not the sum

    # …and growing past the first tenant moves the arena, which is why a tenant
    # that cannot be moved has to say so.
    c = M.reserve!(p, d, :buf, nothing, 4096; blocksize = 4096)
    @test p.arenas[:buf].bytes == 4096
    @test length(c) >= 4096

    x = M.allocate(p, d, :buf, Float32, (64,); blocksize = 4096)
    y = M.allocate(p, d, :buf, Float32, (64,); blocksize = 4096)
    xr, yr = M.region(x), M.region(y)
    @test M.offset(xr) != M.offset(yr)          # private: two buffers, two slices
    lo1, hi1 = M.offset(xr), M.offset(xr) + length(xr)
    lo2, hi2 = M.offset(yr), M.offset(yr) + length(yr)
    @test hi1 <= lo2 || hi2 <= lo1              # …and genuinely disjoint

    # a tenant that refuses to move stops the arena growing, rather than being
    # re-materialised under a recording that names the old addresses
    struct Nailed end
    M.remappable(::Nailed) = false
    M.tenant!(p, :buf, Nailed())
    @test_throws ArgumentError M.reserve!(p, d, :buf, nothing, 1 << 20; blocksize = 4096)
end

# A device whose constraints behave like a real one's: usage bits that must be
# PERMITTED (so they union), against a fake whose kind was its own constraint.
struct BitDev
    allocs::Vector{Int}
end
BitDev() = BitDev(Int[])
M.rawalloc(d::BitDev, kind, bytes, c) = (push!(d.allocs, bytes); zeros(UInt8, bytes))
M.rawfree(::BitDev, mem) = nothing
M.constraintof(::BitDev, kind, ts) = ts === nothing ? UInt32(0) : UInt32(ts)
M.compatible(::BitDev, blk::UInt32, req::UInt32) = (blk & req) == req
M.mergeconstraints(::BitDev, kind, a::UInt32, b::UInt32) = a | b

@testset "an arena reconciles what all its tenants need" begin
    # reserve!'s fast path used to return the existing region on SIZE alone. An
    # arena outlives the plan that sized it, so a block created for one plan's
    # usage bits may not permit what the next plan does with them — memory that
    # works until the one transient whose bit was missing is used.
    d, p = BitDev(), M.Pool()
    r1 = M.reserve!(p, d, :buf, 0b0001, 1024; blocksize = 4096)
    @test p.arenas[:buf].constraint == 0b0001

    # smaller, but needs a bit the block does not have: must NOT reuse it
    r2 = M.reserve!(p, d, :buf, 0b0010, 512; blocksize = 4096)
    @test p.arenas[:buf].constraint == 0b0011        # union, so both are served
    @test M.offset(r2) != M.offset(r1) || r2.block !== r1.block
    @test length(d.allocs) == 2                      # a second block was needed

    # smaller AND already permitted: now the fast path is right to reuse
    before = length(d.allocs)
    r3 = M.reserve!(p, d, :buf, 0b0001, 256; blocksize = 4096)
    @test r3 === p.arenas[:buf].region
    @test length(d.allocs) == before                 # reached no device

    # `nothing` is a CONSTRAINT, not a failure signal. The host backend's is
    # exactly that — its memory is just memory — and using `nothing` to mean
    # "irreconcilable" made every host graph unplaceable. Caught by the editor's
    # fuzz suite, which is the only thing that runs two host plans in one pool.
    struct NilDev end
    M.rawalloc(::NilDev, kind, bytes, c) = zeros(UInt8, bytes)
    M.rawfree(::NilDev, mem) = nothing
    M.constraintof(::NilDev, kind, ts) = nothing
    M.compatible(::NilDev, a, b) = true
    p3 = M.Pool()
    r4 = M.reserve!(p3, NilDev(), :any, nothing, 64; blocksize = 4096)
    r5 = M.reserve!(p3, NilDev(), :any, nothing, 64; blocksize = 4096)
    @test r4 === r5                                  # shared, and neither threw
    @test p3.arenas[:any].constraint === nothing

    # and a device that cannot reconcile says so, instead of handing back memory
    # that satisfies only one of them
    struct PickyDev end
    M.rawalloc(::PickyDev, kind, bytes, c) = zeros(UInt8, bytes)
    M.rawfree(::PickyDev, mem) = nothing
    M.constraintof(::PickyDev, kind, ts) = ts
    M.compatible(::PickyDev, a, b) = a === b
    p2 = M.Pool()
    M.reserve!(p2, PickyDev(), :img, :typeA, 64; blocksize = 4096)
    @test_throws ArgumentError M.reserve!(p2, PickyDev(), :img, :typeB, 64; blocksize = 4096)
end
