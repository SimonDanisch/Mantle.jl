mutable struct CachePublication{V}
    ready::Base.Event
    owner::Union{Nothing,Task}
    # `Some{Any}`, not `Some{V}`: boxed once, so a hit does not rebuild an
    # immutable native aggregate such as a `Vector{LaunchPlan}`.
    value::Union{Nothing,Some{Any}}
    failure::Any
end
CachePublication{V}() where {V} = CachePublication{V}(Base.Event(), current_task(), nothing, nothing)

"""
    CompilationCache{K,V}()

Publishes device-owned compiled objects: one producer per key, unrelated keys
compile concurrently. No lock is held while building or waiting, so finalizers
stay enabled during native creation. A failed build wakes its waiters and is
removed so a later call retries. `empty!` also detaches pending entries: their
callers still get the value, but it does not repopulate the cache.
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
