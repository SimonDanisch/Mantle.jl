using Test, Lava, Mantle
using Lava: size_class, size_class_idx, size_class_bytes,
            POOL_MIN_SIZE, POOL_SUBDIV, POOL_SUBDIV_MIN, POOL_LARGE_THRESHOLD,
            POOL_NUM_SIZE_CLASSES

# The pool hands a chunk out by looking its class up from the REQUEST, and takes
# it back by looking the class up from the SIZE IT HANDED OUT. If those two ever
# disagree a freed chunk lands in a list for a different size, and the next
# caller to pop it is handed a buffer smaller than it asked for — silent
# heap corruption on the GPU. Hence the round-trip property, checked exhaustively
# where it is most likely to break: the octave boundaries.
@testset "pool size classes" begin
    @testset "round trip: class(bytes(n)) == class(n)" begin
        sizes = Int[]
        append!(sizes, 1:4096)                       # the whole power-of-two head
        for p in 12:26                               # every subdivided octave
            base = 1 << p
            step = base >> 3
            for s in 0:POOL_SUBDIV
                for d in (-2, -1, 0, 1, 2)
                    n = base + s * step + d
                    POOL_MIN_SIZE <= n <= POOL_LARGE_THRESHOLD && push!(sizes, n)
                end
            end
        end
        bad = filter(n -> size_class(size_class(n)[2]) != size_class(n), sizes)
        @test isempty(bad)
    end

    @testset "a class holds exactly one size" begin
        # Two requests in the same list must round to the same byte count, or
        # free-list reuse hands back the wrong capacity.
        seen = Dict{Int,Int}()
        conflicts = Tuple{Int,Int,Int}[]
        for n in vcat(collect(1:4096), [(1 << p) + k * ((1 << p) >> 6)
                                        for p in 12:26 for k in 0:63])
            n = clamp(n, POOL_MIN_SIZE, POOL_LARGE_THRESHOLD)
            i, b = size_class(n)
            if haskey(seen, i)
                seen[i] == b || push!(conflicts, (i, seen[i], b))
            else
                seen[i] = b
            end
        end
        @test isempty(conflicts)
    end

    @testset "never returns less than requested" begin
        ns = vcat(collect(1:4096), [rand(4097:POOL_LARGE_THRESHOLD) for _ in 1:20_000])
        @test all(n -> size_class_bytes(n) >= n, ns)
    end

    @testset "waste is bounded by 1/POOL_SUBDIV above the head" begin
        worst = 0.0
        for p in 12:25, s in 0:(POOL_SUBDIV - 1)
            base = 1 << p; step = base >> 3
            n = base + s * step + 1                  # just past a class boundary
            worst = max(worst, size_class_bytes(n) / n - 1)
        end
        # Plain powers of two would give ~100% here.
        @test worst <= 1 / POOL_SUBDIV
        @test worst > 0                              # …and it does round up
    end

    @testset "indices are in range" begin
        ns = vcat(collect(POOL_MIN_SIZE:4096),
                  [rand(4097:POOL_LARGE_THRESHOLD) for _ in 1:20_000],
                  [POOL_LARGE_THRESHOLD])
        idxs = map(size_class_idx, ns)
        @test all(i -> 1 <= i <= POOL_NUM_SIZE_CLASSES, idxs)
    end

    @testset "power-of-two head is unchanged" begin
        # The old scheme, for every size it used to cover.
        @test all(n -> size_class(n) == (trailing_zeros(nextpow(2, max(n, POOL_MIN_SIZE))) - 3,
                                         nextpow(2, max(n, POOL_MIN_SIZE))),
                  1:POOL_SUBDIV_MIN)
    end

    @testset "alignment: every class size is a multiple of 16" begin
        ns = vcat(collect(POOL_MIN_SIZE:4096), [rand(4097:POOL_LARGE_THRESHOLD) for _ in 1:20_000])
        @test all(n -> size_class_bytes(n) % 16 == 0, ns)
    end
end
