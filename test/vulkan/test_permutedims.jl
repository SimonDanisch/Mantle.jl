"""
`permutedims!` on a LavaArray: the right answer first, the throughput second.

Written after a version that used `perm` where it needed `invperm(perm)` passed a
throughput benchmark — because the shape being benchmarked, `(1,3,2,4)`, is
self-inverse — and then produced garbage for SAM 2, whose encoder also uses
permutations that are not. Every case here is checked against Base.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "permutedims! on LavaArray" begin
    back = LavaBackend()

    @testset "matches Base, including non-self-inverse permutations" begin
        cases = (((4, 6, 8), (2, 1, 3)),        # self-inverse
                 ((4, 6, 8), (3, 1, 2)),        # NOT self-inverse — the one that caught it
                 ((4, 6, 8), (2, 3, 1)),        # its inverse
                 ((3, 5), (2, 1)),
                 ((72, 8, 4, 6), (1, 3, 2, 4)), # the shape attention actually uses
                 ((2, 3, 4, 5), (4, 3, 2, 1)),
                 ((2, 3, 4, 5), (3, 1, 4, 2)),
                 # Rank-4 shapes big enough to need more than one workgroup along
                 # two axes at once. These were silently 7/8 unwritten while the
                 # launch used `kernel(backend, wg)`; see test_static_workgroup.jl.
                 ((256, 72, 8, 16), (2, 1, 3, 4)),
                 ((256, 72, 8, 16), (3, 1, 2, 4)),
                 ((72, 256, 4, 4), (2, 1, 4, 3)))
        for (dims, perm) in cases
            Ah = reshape(collect(Float32, 1:prod(dims)), dims)
            want = permutedims(Ah, perm)
            a = KA.allocate(back, Float32, dims...); copyto!(a, Ah)
            b = KA.allocate(back, Float32, size(want)...)
            permutedims!(b, a, perm)
            KA.synchronize(back)
            @test Array(b) == want
            a = b = nothing; GC.gc()
        end
    end

    @testset "accepts a vector perm and rejects bad ones" begin
        a = KA.allocate(back, Float32, 4, 6); copyto!(a, reshape(collect(Float32, 1:24), 4, 6))
        b = KA.allocate(back, Float32, 6, 4)
        permutedims!(b, a, [2, 1])                       # AbstractVector, not a tuple
        KA.synchronize(back)
        @test Array(b) == permutedims(reshape(collect(Float32, 1:24), 4, 6), (2, 1))
        @test_throws ArgumentError permutedims!(b, a, (1, 1))       # not a permutation
        @test_throws ArgumentError permutedims!(b, a, (2, 1, 3))    # wrong length
        bad = KA.allocate(back, Float32, 4, 6)
        @test_throws DimensionMismatch permutedims!(bad, a, (2, 1))  # wrong destination shape
        a = b = bad = nothing; GC.gc()
    end

    @testset "eltypes other than Float32" begin
        for T in (Float16, Int32)
            Ah = reshape(collect(T, 1:24), 2, 3, 4)
            want = permutedims(Ah, (3, 1, 2))
            a = KA.allocate(back, T, 2, 3, 4); copyto!(a, Ah)
            b = KA.allocate(back, T, size(want)...)
            permutedims!(b, a, (3, 1, 2)); KA.synchronize(back)
            @test Array(b) == want
            a = b = nothing; GC.gc()
        end
    end
end
