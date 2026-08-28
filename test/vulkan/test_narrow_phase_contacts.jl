using Test, Lava, KernelAbstractions
using Mantle: UnitCube, ContactRecord, narrow_phase_contacts_kernel
using GeometryBasics: Vec3f
using KernelAbstractions: CPU

# Shared narrow-phase helpers — see test/narrow_phase_helpers.jl.
isdefined(@__MODULE__, :tx) ||
    include(joinpath(@__DIR__, "narrow_phase_helpers.jl"))

# Run the fused kernel on the KA.CPU backend.  `n_grains` is the number of
# grains; the contact buffer is sized `n_grains * max_contacts`.  Returns
# (counters, contacts).
function run_cpu_compact(transforms, pairs, shape, n_grains, max_contacts)
    counters = zeros(UInt32, n_grains)
    sentinel = ContactRecord(typemax(UInt32), typemax(UInt32),
                             Vec3f(0f0, 0f0, 0f0),
                             Vec3f(0f0, 0f0, 0f0),
                             0f0)
    contacts = fill(sentinel, n_grains * max_contacts)
    narrow_phase_contacts_kernel(CPU())(transforms, pairs, shape,
                                        counters, contacts,
                                        Int32(max_contacts);
                                        ndrange = length(pairs))
    KernelAbstractions.synchronize(CPU())
    return counters, contacts
end

# Pull the live records for grain g into a small Vector — slot indices are
# 1..min(counter[g], max_contacts).
function records_for(g::Integer, counters, contacts, max_contacts)
    n = min(Int(counters[g]), max_contacts)
    base = (Int(g) - 1) * max_contacts
    return contacts[(base + 1):(base + n)]
end

@testset "narrow_phase_contacts_kernel — KA.CPU" begin

    @testset "single overlapping pair lands in BOTH grain slot lists" begin
        # Two unit cubes overlapping by 0.1 along +X.
        transforms   = [tx(0,0,0), tx(1.9, 0, 0)]
        pairs        = [(Int32(1), Int32(2))]
        max_contacts = 4
        n_grains     = 2

        counters, contacts = run_cpu_compact(transforms, pairs, UnitCube(),
                                             n_grains, max_contacts)

        # Each grain gets exactly one contact.
        @test counters[1] == UInt32(1)
        @test counters[2] == UInt32(1)

        recs1 = records_for(1, counters, contacts, max_contacts)
        recs2 = records_for(2, counters, contacts, max_contacts)
        @test length(recs1) == 1
        @test length(recs2) == 1

        # Both copies carry the same (i, j) pair indices and same EPA fields.
        for r in (recs1[1], recs2[1])
            @test r.i == UInt32(1)
            @test r.j == UInt32(2)
            @test r.depth ≈ 0.1f0 atol=1f-3
            @test r.n_hat[1] ≈ 1f0 atol=1f-3
            @test abs(r.n_hat[2]) < 1f-3
            @test abs(r.n_hat[3]) < 1f-3
            @test r.p[1] ≈ 1f0 atol=1f-2
        end
    end

    @testset "separated pair writes nothing; counters stay at 0" begin
        transforms   = [tx(0,0,0), tx(5, 0, 0)]
        pairs        = [(Int32(1), Int32(2))]
        max_contacts = 4
        n_grains     = 2

        counters, contacts = run_cpu_compact(transforms, pairs, UnitCube(),
                                             n_grains, max_contacts)

        @test counters == UInt32[0, 0]
        # Sentinel slots untouched.
        @test all(c.i == typemax(UInt32) for c in contacts)
    end

    @testset "4-cube square: every grain ends up with 3 contacts" begin
        # Same geometry as the narrow_phase_kernel batch-of-6 test: 4 cubes at
        # corners of a 1.9-side square; all 6 C(4,2) pairs overlap.
        transforms = [tx(0,0,0), tx(1.9,0,0), tx(0,1.9,0), tx(1.9,1.9,0)]
        pairs = [
            (Int32(1), Int32(2)),
            (Int32(1), Int32(3)),
            (Int32(2), Int32(4)),
            (Int32(3), Int32(4)),
            (Int32(1), Int32(4)),
            (Int32(2), Int32(3)),
        ]
        max_contacts = 8
        n_grains     = 4

        counters, contacts = run_cpu_compact(transforms, pairs, UnitCube(),
                                             n_grains, max_contacts)

        # Each grain participates in exactly 3 pairs.
        @test counters == UInt32[3, 3, 3, 3]

        # Each grain's 3 records reference the right peer indices.
        peers = Dict(
            1 => Set([2, 3, 4]),
            2 => Set([1, 3, 4]),
            3 => Set([1, 2, 4]),
            4 => Set([1, 2, 3]),
        )
        for g in 1:4
            recs = records_for(g, counters, contacts, max_contacts)
            seen = Set{Int}()
            for r in recs
                # The "other" grain is whichever of (i, j) isn't `g`.
                other = Int(r.i) == g ? Int(r.j) : Int(r.i)
                push!(seen, other)
                @test r.depth > 0f0
            end
            @test seen == peers[g]
        end
    end

    @testset "overflow: counter saturates, no out-of-bounds writes" begin
        # 5 pairs all touching grain 1; max_contacts = 3 → grain 1 loses 2.
        transforms = [tx(0,0,0),
                      tx(1.9, 0, 0),
                      tx(0, 1.9, 0),
                      tx(0, 0, 1.9),
                      tx(-1.9, 0, 0),
                      tx(0, -1.9, 0)]
        pairs = [(Int32(1), Int32(k)) for k in 2:6]   # 5 pairs all involve grain 1
        max_contacts = 3
        n_grains     = 6

        counters, contacts = run_cpu_compact(transforms, pairs, UnitCube(),
                                             n_grains, max_contacts)

        # All 5 pair threads atomically incremented counter[1] once each.
        @test counters[1] == UInt32(5)
        # Grains 2..6 each only saw one pair → counter == 1.
        @test all(counters[2:6] .== UInt32(1))

        # Only the first 3 slots of grain 1's list got writes; slots 4..5 are
        # outside the per-grain range and must not have been touched.  Verify
        # by reading directly into grain 1's range.
        live_for_1 = contacts[1:max_contacts]
        @test all(r.i == UInt32(1) for r in live_for_1)
        @test all(r.depth > 0f0 for r in live_for_1)

        # Verify grain 2..6's first slots got their copy.
        for g in 2:6
            slot = (g - 1) * max_contacts + 1
            @test contacts[slot].depth > 0f0
            @test (Int(contacts[slot].i), Int(contacts[slot].j)) == (1, g)
        end
    end
end

# ---------------------------------------------------------------------------
# GPU smoke test on the Lava backend.
#
# Per the project "never crash-loop GPU tests" rule, this is a single
# minimal run (one overlapping pair) and any failure is investigated, not
# retried.  Mirrors the CPU "single overlapping pair lands in BOTH grain
# slot lists" testset to confirm GPU compaction matches CPU.
# ---------------------------------------------------------------------------
@testset "narrow_phase_contacts_kernel — Lava backend (GPU smoke)" begin
    using Mantle: LavaArray, LavaBackend, ContactRecord, narrow_phase_contacts_kernel
    using GeometryBasics: Vec3f
    max_contacts = Int32(4)
    n_grains     = 2
    transforms = LavaArray([tx(0,0,0), tx(1.9, 0, 0)])
    pairs      = LavaArray([(Int32(1), Int32(2))])
    counters   = LavaArray(zeros(UInt32, n_grains))
    sentinel   = ContactRecord(typemax(UInt32), typemax(UInt32),
                               Vec3f(0f0, 0f0, 0f0),
                               Vec3f(0f0, 0f0, 0f0),
                               0f0)
    contacts   = LavaArray(fill(sentinel, n_grains * Int(max_contacts)))

    narrow_phase_contacts_kernel(LavaBackend())(
        transforms, pairs, Mantle.UnitCube(),
        counters, contacts, max_contacts;
        ndrange = 1)
    Mantle.vk_flush!(Mantle.vk_context().default_bq)

    cs = Array(counters)
    rs = Array(contacts)

    @test cs[1] == UInt32(1)
    @test cs[2] == UInt32(1)

    # Slot 1 of each grain holds the contact record; both copies carry
    # the same (i, j) pair indices.
    for r in (rs[1], rs[max_contacts + 1])
        @test r.i == UInt32(1)
        @test r.j == UInt32(2)
        @test r.depth ≈ 0.1f0 atol=1f-3
        @test r.n_hat[1] ≈ 1f0 atol=1f-3
        @test abs(r.n_hat[2]) < 1f-3
        @test abs(r.n_hat[3]) < 1f-3
    end

    # Untouched slots still hold the sentinel.
    for k in (2, 3, 4, max_contacts + 2, max_contacts + 3, max_contacts + 4)
        @test rs[k].i == typemax(UInt32)
    end
end

