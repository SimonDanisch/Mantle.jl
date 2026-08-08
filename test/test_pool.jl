# The suballocator, exercised with no backend whatsoever.
#
# That this file needs neither Lava nor KernelAbstractions is the assertion: if
# `Pool` ever needs a real device to be tested, some policy has leaked into a
# backend and a second backend will have to copy it.

using Test
import Mantle
const M = Mantle

"""
A device that allocates nothing and counts what it was asked for.

`allocs` is the whole point of the fixture — the headline property is "a second
acquire does not reach the device", and that is only observable by counting.
"""
struct FakeDev
    allocs::Vector{Int}
end
FakeDev() = FakeDev(Int[])

M.rawalloc(d::FakeDev, kind, bytes, c) = (push!(d.allocs, bytes); zeros(UInt8, bytes))
M.rawfree(::FakeDev, mem) = nothing
M.constraintof(::FakeDev, kind, ts) = kind          # kind IS the constraint here
M.compatible(::FakeDev, a, b) = a === b

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
M.upload!(::FakeDev, a::M.DeviceArray{T}, first, data) where {T} =
    (v = reinterpret(T, view(M.memoryof(a), (M.offset(a)+1):(M.offset(a)+sizeof(a))));
     copyto!(v, first, data, 1, length(data)); a)
M.download(::FakeDev, a::M.DeviceArray{T}) where {T} =
    collect(reinterpret(T, view(M.memoryof(a), (M.offset(a)+1):(M.offset(a)+sizeof(a)))))
M.devicecopy!(d::FakeDev, dst, src, n) =
    (M.upload!(d, dst, 1, M.download(d, src)[1:n]); dst)
M.deviceview(::FakeDev, a) = a

@testset "Buffer/Scalar are core types over pool regions" begin
    d = FakeDev(); p = M.Pool(); dev = d
    # the pool has to be reachable from the device, as it is for a real one
    @eval M.pool(::$(typeof(d))) = $p

    b = M.Buffer(dev, Float32[1, 2, 3]; capacity = 8)
    @test length(b) == 3 && M.capacity(b) == 8 && eltype(b) == Float32
    @test Array(b) == Float32[1, 2, 3]
    @test length(d.allocs) == 1                     # ONE block, not one per resource

    s = M.Scalar(dev, 7.0f0)
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
