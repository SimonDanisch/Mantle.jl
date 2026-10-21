mutable struct CachePublication{V}
    ready::Base.Event
    owner::Union{Nothing,Task}
    # Keep immutable native aggregates boxed once too: an inline Some{V}
    # can materialize a new aggregate on every hit, like Vector{LaunchPlan}.
    value::Union{Nothing,Some{Any}}
    failure::Any
end
CachePublication{V}() where {V} = CachePublication{V}(Base.Event(), current_task(), nothing, nothing)

"""
    CompilationCache{K,V}()

Core publication of device-owned compiled objects. One producer runs per key;
unrelated keys can compile concurrently. No map lock is held while compiling,
allocating or waiting, so GC finalizers remain enabled during native creation.
Failed publications wake their waiters and are removed, allowing a later retry.

`empty!` detaches pending publications as well as ready ones. Existing callers
can finish with their publication, but it cannot repopulate the cleared cache.
This coordinates publication, not idle retention or resource retirement.
"""
struct CompilationCache{K,V}
    lock::ReentrantLock
    entries::Dict{K,CachePublication{V}}
end
CompilationCache{K,V}() where {K,V} =
    CompilationCache{K,V}(ReentrantLock(), Dict{K,CachePublication{V}}())

function cachevalue(entry::CachePublication{V}) where {V}
    wait(entry.ready) # acquire/release ordering publishes value or failure
    value = entry.value
    value === nothing && throw(entry.failure)
    return something(value)::V
end

function Base.get!(build, cache::CompilationCache{K,V}, key) where {K,V}
    entry, producer = lock(cache.lock) do
        found = get(cache.entries, key, nothing)
        if found === nothing
            fresh = CachePublication{V}()
            cache.entries[key] = fresh
            return fresh, true
        end
        found.owner === current_task() && throw(ArgumentError("recursive compilation of the same cache key"))
        return found, false
    end
    producer || return cachevalue(entry)
    try
        value = Some{Any}(convert(V, build()))
        lock(cache.lock) do
            entry.value = value
            entry.owner = nothing
        end
        return something(value)::V
    catch err
        lock(cache.lock) do
            entry.value = nothing
            entry.failure = err
            entry.owner = nothing
            get(cache.entries, key, nothing) === entry && delete!(cache.entries, key)
        end
        rethrow()
    finally
        notify(entry.ready)
    end
end

function Base.getindex(cache::CompilationCache, key)
    entry = lock(cache.lock) do
        found = cache.entries[key]
        found.owner === current_task() && throw(ArgumentError("recursive compilation of the same cache key"))
        return found
    end
    return cachevalue(entry)
end

Base.empty!(cache::CompilationCache) = (lock(() -> empty!(cache.entries), cache.lock); cache)
Base.length(cache::CompilationCache) = lock(() -> length(cache.entries), cache.lock)

# Keep code identities alive while they are keys. Bare objectid integers can
# outlive their objects and alias an unrelated object after address reuse.
struct CacheIdentity{T}
    value::T
end
Base.isequal(a::CacheIdentity, b::CacheIdentity) = a.value === b.value
Base.hash(a::CacheIdentity, h::UInt) = hash(objectid(a.value), h)
