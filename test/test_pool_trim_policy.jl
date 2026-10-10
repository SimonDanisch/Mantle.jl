# When a pool trims and collects ON ITS OWN, without being asked: its policy, on
# every backend. `makeroom!`, the explicit trim, is `test_pool_trim.jl`'s.
#
# Empty pool blocks must be returned to the driver without waiting for an
# out-of-memory. Without a threshold of their own they only went back on an
# allocation-failure retry, and the pressure gate is a ratio against the device
# heap: on an iGPU with a large shared heap a few GB of dead blocks is ~20 % and
# never trips it, so the pool ratcheted up to the high-water mark of the largest
# workload and stayed there. Across a multi-scene run that is gigabytes of system
# RAM the rest of the machine still needs, and it showed up as a driver timeout
# partway through a 5-scene sweep. `autotrim!` releases dead capacity on its own
# terms, by `pool(dev).policy`.
#
# This was the Vulkan backend's policy for its pooled arrays, applied in
# `pool_alloc`; Metal's pool had none, so a Mac held every block it ever grew. It
# is the pool's now, and every backend's pool applies it to what it allocates —
# here through `Buffer`s, which are pool regions on every backend.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""
Allocate 16 MB `Buffer`s on `dev` until its pool has reserved `target` bytes, and
drop them all. In a function of its own, so the vector holding them is garbage
when it returns: emptying a vector the caller still holds leaves its memory
naming the buffers, and nine of them survived every collection that way.
"""
function growanddrop!(dev, p, target)
    bufs = Mantle.Buffer[]
    while Mantle.reserved(p) < target
        push!(bufs, Mantle.Buffer(dev, Float32, (4_000_000,)))
    end
    return length(bufs)
end

"""Run `f` with the pool's policy as it is, and put every field back after: the
policy is the device's, and the next file inherits it."""
function withpolicy(f, p::Mantle.Pool)
    was = [getfield(p.policy, k) for k in fieldnames(Mantle.PoolPolicy)]
    try
        f(p.policy)
    finally
        for (k, v) in zip(fieldnames(Mantle.PoolPolicy), was)
            setfield!(p.policy, k, v)
        end
    end
end

@testset "empty pool blocks are trimmed without an out-of-memory" begin
    dev = Mantle.Device(TESTBACKEND)
    p = Mantle.pool(dev)
    withpolicy(p) do policy
        # A threshold this test can reach in a few hundred MB: the threshold is
        # configuration, and the default (1 GiB) is the device's to keep.
        policy.trim_threshold = 128 << 20
        # The allocations below run the policy themselves; a trim just now keeps
        # them from trimming the blocks this test is growing.
        policy.last_trim = time()
        # Against this file's own baseline, not zero: other files' blocks are in
        # the pool too, and what is checked is that the capacity THIS creates
        # comes back.
        base = Mantle.reserved(p)
        # Dropped, not freed: the trim's own collection finds them.
        @test growanddrop!(dev, p, base + policy.trim_threshold + (64 << 20)) > 0
        grown = Mantle.reserved(p)
        @test grown - base >= policy.trim_threshold
        Mantle.waitidle(dev)

        # Rate limited: a trim moments ago means none now.
        policy.last_trim = time()
        @test !Mantle.autotrim!(p, dev)
        @test Mantle.reserved(p) == grown

        # Past the interval it runs, and the capacity comes back. With the full
        # collection's own interval passed too, so the trim may pay for one if the
        # incremental collection did not reach the dropped buffers.
        policy.last_trim = 0.0
        policy.last_full_gc = 0.0
        @test Mantle.autotrim!(p, dev)
        trimmed = Mantle.reserved(p)
        @test trimmed < grown                     # capacity actually came back
        @test trimmed - base < policy.trim_threshold   # and it was THIS testset's

        # Under the threshold, nothing to do.
        policy.last_trim = 0.0
        policy.trim_threshold = typemax(Int)
        @test !Mantle.autotrim!(p, dev)
    end
    # And the allocator still works afterwards — blocks were returned, not corrupted.
    b = Mantle.Buffer(dev, fill(2f0, 1024))
    Mantle.waitidle(dev)
    @test all(==(2f0), Array(Mantle.storage(b)))
end

# The other rate limit, and the one that was missing: a soft-cap collection is
# spaced by what it COSTS, not only by how often it may run.
#
# `gc_mingap` is 20 ms. An incremental collection on a heap holding a couple of
# GiB of device arrays takes ~63 ms, so the gap alone permitted spending three
# quarters of the clock inside the allocator — and it did: RIFE at 1920x1152 sits
# at 2094.6 MiB against the 2048 MiB default cap, collected six times a run, and
# 142 ms of work was reported as a p50 of 519 ms with a spread out to 912. The
# pool was 2% over the cap the whole time, so there was nothing to win either.
@testset "a soft-cap collection is limited by what it cost" begin
    p = Mantle.pool(Mantle.Device(TESTBACKEND))
    drains = Ref(0)
    drain() = (drains[] += 1; nothing)
    withpolicy(p) do policy
        # A collection that cost 100 ms, on a 5% budget, is a 2 s gap — so one
        # 20 ms after it is refused, where `gc_mingap` alone would allow it.
        policy.gc_budget = 0.05
        policy.gc_lastcost = 0.1
        policy.gc_last = time() - 0.02
        @test !Mantle.autocollect!(drain, p)
        @test drains[] == 0
        # Past the cost-derived gap it runs again, and drains what it freed.
        policy.gc_last = time() - 2.5
        @test Mantle.autocollect!(drain, p)
        @test drains[] >= 1
        # …and what it just cost is remembered, which is what spaces the next one.
        @test policy.gc_lastcost > 0.0
        # `gc_mingap` still applies when a collection was cheap: the budget can
        # only ever make the gap LONGER, never shorter.
        policy.gc_lastcost = 0.0
        policy.gc_last = time()
        @test !Mantle.autocollect!(drain, p)
        # A budget of 1 is no bound at all, which is what this did before.
        policy.gc_budget = 1.0
        policy.gc_lastcost = 0.1
        policy.gc_last = time() - 0.15
        @test Mantle.autocollect!(drain, p)
    end
end
