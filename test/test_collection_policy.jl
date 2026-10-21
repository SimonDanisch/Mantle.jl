using Test, Mantle

@testset "one collection budget for heap pressure and soft cap" begin
    p = Mantle.Pool()
    policy = p.policy
    policy.gc_mingap = 3600.0
    policy.gc_full_mingap = Inf
    drains = Ref(0)
    drain() = (drains[] += 1)
    @test !Mantle.autocollect!(drain, p; pressure=0.74)
    @test drains[] == 0
    @test Mantle.autocollect!(drain, p; pressure=0.8)
    @test drains[] == 1
    @test policy.gc_count == 1
    # A second entry point cannot spend a separate budget on the same pool.
    @test !Mantle.autocollect!(drain, p)
    @test drains[] == 1
    @test policy.gc_count == 1

    other = Mantle.Pool()
    other.policy.gc_full_mingap = Inf
    @test Mantle.autocollect!(drain, other)
    @test other.policy.gc_count == 1
    @test policy.gc_count == 1

    # An urgent trigger still respects the cost of the preceding collection.
    policy.gc_lastcost = 1.0
    policy.gc_last = time() - 0.1
    policy.gc_mingap = 0.0
    @test !Mantle.autocollect!(drain, p; pressure=0.99, blocking=true)
end

@testset "collection cannot re-enter through its drain" begin
    p = Mantle.Pool()
    p.policy.gc_full_mingap = Inf
    nested = Ref(true)
    trimmed = Ref(true)
    @test Mantle.autocollect!(p) do
        nested[] = Mantle.autocollect!(() -> error("nested collection ran"), p; pressure=1.0)
        p.policy.trim_threshold = 0
        trimmed[] = Mantle.autotrim!(p, nothing)
        Mantle.makeroom!(p, nothing)
    end
    @test !nested[]
    @test !trimmed[]
    @test p.policy.gc_count == 1
end

@testset "trimming shares the last full collection" begin
    p = Mantle.Pool()
    p.policy.gc_full_mingap = 0.0
    @test Mantle.autocollect!(() -> nothing, p)
    last_full = p.policy.gc_full_last
    @test last_full > 0.0
    p.policy.trim_threshold = 0
    @test !Mantle.autotrim!(p, nothing)
    @test p.policy.gc_full_last == last_full
end

@testset "a failed drain releases the claim and charges its collection" begin
    p = Mantle.Pool()
    p.policy.gc_full_mingap = Inf
    @test_throws ErrorException("drain failed") Mantle.autocollect!(() -> error("drain failed"), p)
    @test p.policy.gc_count == 1
    @test p.policy.gc_lastcost > 0
    p.policy.gc_last = 0.0
    @test Mantle.autocollect!(() -> nothing, p)
    @test p.policy.gc_count == 2
end

@testset "a concurrent entry cannot run a second collection" begin
    p = Mantle.Pool()
    p.policy.gc_full_mingap = Inf
    entered, proceed = Channel{Nothing}(1), Channel{Nothing}(1)
    first = @async Mantle.autocollect!(p) do
        put!(entered, nothing)
        take!(proceed)
    end
    take!(entered)
    try
        @test !Mantle.autocollect!(() -> error("concurrent collection ran"), p)
    finally
        put!(proceed, nothing)
    end
    @test fetch(first)
    @test p.policy.gc_count == 1
end

@testset "a paced collection rejection allocates nothing" begin
    p = Mantle.Pool()
    p.policy.gc_last = time()
    p.policy.gc_mingap = 3600.0
    rejectedbytes(p) = @allocated Mantle.autocollect!(() -> nothing, p)
    rejectedbytes(p)
    @test rejectedbytes(p) == 0
end
