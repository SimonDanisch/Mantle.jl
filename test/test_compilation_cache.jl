using Test, Mantle

@testset "compiled objects have one publication per key" begin
    cache = Mantle.CompilationCache{Int,Vector{Int}}()
    builds = Threads.Atomic{Int}(0)
    jobs = [Threads.@spawn get!(cache, 1) do
        Threads.atomic_add!(builds, 1)
        yield()
        [42]
    end for _ in 1:32]
    values = fetch.(jobs)
    @test builds[] == 1
    @test all(v -> v === values[1], values)
    @test values[1] == [42]
    @test cache[1] === values[1]
    @test length(cache) == 1
end

mutable struct CollidingCode
    bytes::Vector{UInt8}
end
Base.objectid(::CollidingCode) = UInt(0) # force the hash collision, not an equality

@testset "code identity distinguishes equal bytes and colliding hashes" begin
    a, b = CollidingCode(UInt8[1]), CollidingCode(UInt8[1])
    cache = Mantle.CompilationCache{Tuple,Int}()
    ka, kb = (Mantle.CacheIdentity(a),), (Mantle.CacheIdentity(b),)
    @test hash(ka) == hash(kb)
    @test !isequal(ka, kb)
    @test get!(() -> 1, cache, ka) == 1
    @test get!(() -> 2, cache, kb) == 2
    @test get!(() -> error("lost code identity"), cache, (Mantle.CacheIdentity(a),)) == 1
end

@testset "one pending compile does not block unrelated keys" begin
    cache = Mantle.CompilationCache{Int,Int}()
    entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
    first = Threads.@spawn get!(cache, 1) do
        put!(entered, nothing)
        take!(release)
        11
    end
    take!(entered)
    second = Threads.@spawn get!(() -> 22, cache, 2)
    try
        @test timedwait(() -> istaskdone(second), 5.0) === :ok
    finally
        put!(release, nothing)
    end
    @test fetch(first) == 11
    @test fetch(second) == 22
end

@testset "failed publication wakes waiters and permits a later retry" begin
    cache = Mantle.CompilationCache{Int,Int}()
    entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
    failure = ErrorException("native creation refused")
    capture(f) = try f() catch err; err end
    first = @async capture() do
        get!(cache, 1) do
            put!(entered, nothing)
            take!(release)
            throw(failure)
        end
    end
    take!(entered)
    waiters = Task[]
    for _ in 1:8
        push!(waiters, @async capture(() -> get!(() -> error("duplicate producer"), cache, 1)))
        yield() # cooperative waiter reaches the pending publication
    end
    put!(release, nothing)
    @test fetch(first) === failure
    @test all(t -> fetch(t) === failure, waiters)
    @test length(cache) == 0
    @test get!(() -> 7, cache, 1) == 7
    @test cache[1] == 7
end

@testset "failure of a detached compile leaves its replacement intact" begin
    cache = Mantle.CompilationCache{Int,Int}()
    entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
    failure = ErrorException("old creation failed")
    old = @async try
        get!(cache, 1) do
            put!(entered, nothing)
            take!(release)
            throw(failure)
        end
    catch err
        err
    end
    take!(entered)
    empty!(cache)
    @test get!(() -> 2, cache, 1) == 2
    put!(release, nothing)
    @test fetch(old) === failure
    @test cache[1] == 2
end

@testset "clear detaches an in-flight publication" begin
    cache = Mantle.CompilationCache{Int,Vector{Int}}()
    entered, release = Channel{Nothing}(1), Channel{Nothing}(1)
    old = @async get!(cache, 1) do
        put!(entered, nothing)
        take!(release)
        [1]
    end
    take!(entered)
    @test empty!(cache) === cache
    @test length(cache) == 0
    replacement = get!(() -> [2], cache, 1)
    put!(release, nothing)
    @test fetch(old) == [1]
    @test cache[1] === replacement
    @test replacement == [2]
end

@testset "recursive publication fails rather than waiting on itself" begin
    cache = Mantle.CompilationCache{Int,Int}()
    @test_throws ArgumentError get!(() -> get!(() -> 2, cache, 1), cache, 1)
    @test_throws ArgumentError get!(() -> cache[1], cache, 1)
    @test get!(() -> 3, cache, 1) == 3
end

struct CachedAggregate
    left::Vector{Int}
    right::Vector{Int}
end
@testset "a ready aggregate is reused without allocation" begin
    cache = Mantle.CompilationCache{Int,CachedAggregate}()
    value = get!(() -> CachedAggregate([1], [2]), cache, 1)
    hit(c) = get!(() -> error("unexpected cache miss"), c, 1)
    bytes(c) = @allocated hit(c)
    bytes(cache)
    @test hit(cache) === value
    @test bytes(cache) == 0
    @test get!(() -> nothing, Mantle.CompilationCache{Int,Nothing}(), 1) === nothing
end

Base.@noinline function finalizer_probe(counter)
    owner = Ref(0)
    finalizer(_ -> (counter[] += 1), owner)
    return WeakRef(owner)
end
function check_factory_finalizers()
    count = Ref(0)
    weak = finalizer_probe(count)
    cache = Mantle.CompilationCache{Int,Int}()
    result = get!(cache, 1) do
        GC.gc(true)
        count[]
    end
    @test weak.value === nothing
    @test result == 1
end
@testset "native creation leaves finalizers enabled" begin
    check_factory_finalizers()
end
