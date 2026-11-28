# Compute encoder calls Metal.jl does not wrap, the completion query a quiet queue
# needs, and the residency set Mantle keeps per device.
#
# No unmanaged command buffer type: a command buffer committed through Metal.jl
# (`MTL.commit!`, `MTL.commit_with_queue_key!`) is stored in its submission record,
# which reads its status later and holds no reference of its own. An unmanaged
# wrapper released after the commit leaves that record a dangling pointer, so the
# command buffers Mantle commits are Metal.jl's managed `MTL.MTLCommandBuffer`.

"""
    dispatch_threadgroups!(enc, buf, offset, threads)

Dispatch with the threadgroup count read from `buf` on the device: three `UInt32` at
`offset`, the grid in x, y and z (`MTLDispatchThreadgroupsIndirectArguments`). The
count is in threadgroups, not threads, and a zero runs nothing. The command processor
reads it, so `buf` has to be declared to the encoder (`useResources:`) like any other
buffer the dispatch reaches.
"""
function dispatch_threadgroups!(enc::MTL.MTLComputeCommandEncoderLike, buf::MTL.MTLBuffer,
                                offset::Integer, threads::MTL.MTLSize)
    @objc [enc::id{MTL.MTLComputeCommandEncoder} dispatchThreadgroupsWithIndirectBuffer:buf::id{MTL.MTLBuffer}
                                                 indirectBufferOffset:offset::UInt64
                                                 threadsPerThreadgroup:threads::MTL.MTLSize]::Nothing
end

"""
    use_resources!(enc, ids::Vector{id{MTLBuffer}}, usage)

`useResources:count:usage:` over object pointers the caller converted once. Converting
a `Vector{MTLBuffer}` builds a fresh pointer array on every call, which a resource list
that is stable across frames does not need. The caller keeps the buffers alive.
"""
function use_resources!(enc::MTL.MTLComputeCommandEncoderLike, ids::Vector{id{MTL.MTLBuffer}},
                        usage::MTL.MTLResourceUsage)
    @objc [enc::id{MTL.MTLComputeCommandEncoder} useResources:ids::Ptr{id{MTL.MTLBuffer}}
                                                 count:length(ids)::UInt64
                                                 usage:usage::MTL.MTLResourceUsage]::Nothing
end

"""
    memory_barrier!(enc, scope)

Make the encoder's later dispatches wait for the writes of its earlier ones to the
resources in `scope` (`memoryBarrierWithScope:`).
"""
memory_barrier!(enc::MTL.MTLComputeCommandEncoderLike, scope::MTL.MTLBarrierScope) =
    @objc [enc::id{MTL.MTLComputeCommandEncoder} memoryBarrierWithScope:scope::MTL.MTLBarrierScope]::Nothing

"""
    all_completed(queue::MTLCommandQueue) -> Bool

Whether every command buffer committed to `queue` through Metal.jl has completed. A
queue runs in commit order, so this is the status of the last commit. The status is
read inside `MTL.is_completed`, which returns a `Bool`: reading `.status` off the
abstractly typed record here would return a boxed enum from a dynamic call.
"""
function all_completed(queue::MTL.MTLCommandQueue)
    last = MTL.last_committed(queue)
    return last === nothing || MTL.is_completed(last)
end

# An allocation whose owner freed it, which leaves the set once the device has finished
# the work submitted before (`forget_drained!`), in a list that needs no lock to extend.
struct Retired
    allocation::Any
    # The membership it removes, or 0 for whatever membership the allocation has.
    generation::Int
    next::Union{Nothing,Retired}
end

# One allocation a finalizer asked to remove, in a list that needs no lock to extend.
struct Dropped
    allocation::Any
    # The membership it removes, or 0 for whatever membership the allocation has.
    generation::Int
    next::Union{Nothing,Dropped}
end

"""
    Residency(mtldevice)

A residency set Mantle owns, one per device, attached to the device's queues
([`attach!`](@ref)) and declared by each Metal 4 command buffer. A recorded command, an
argument table and a ray query reach memory by address or by `gpuResourceID`, and an
allocation reached that way that is in no residency set is not mapped: reads return
zeros and writes are dropped, with no error.

The set holds a strong reference to every allocation in it, so membership has to end
when the allocation's owner lets go of it. An allocation stays in the set until
[`forget!`](@ref) is called for it, or until the wrapper it was added through is
collected. A finalizer must not wait for a lock, so a removal it asks for while `lock`
is taken is applied by the next call that takes it.

On a device without residency sets (a virtual GPU) `set` is `nothing` and every call
does nothing.
"""
mutable struct Residency
    const set::Union{Nothing,MTL.MTLResidencySet}
    const lock::ReentrantLock
    # The address of each allocation in `set`, with the generation it was added in.
    # The set keeps the allocation alive, so its address is not reused while it is a
    # member; the generation tells a finalizer of a collected wrapper apart from a
    # later allocation at the same address.
    const members::Dict{UInt,Int}
    generation::Int
    @atomic dropped::Union{Nothing,Dropped}
    @atomic retired::Union{Nothing,Retired}
end

function Residency(dev::MTL.MTLDevice)
    # Metal.jl's own condition for residency sets (`can_use_residency_sets`).
    MTL.is_virtual(dev) && return Residency(nothing, ReentrantLock(), Dict{UInt,Int}(), 0, nothing, nothing)
    desc = MTL.MTLResidencySetDescriptor()
    desc.label = "Mantle"
    return Residency(MTL.MTLResidencySet(dev, desc), ReentrantLock(), Dict{UInt,Int}(), 0, nothing, nothing)
end

"""Whether the device has a residency set, without which nothing reached by address is resident."""
has_residency_set(r::Residency) = r.set !== nothing

"""
    attach!(r, queue) -> r

Make every command buffer of `queue` use `r`'s set, including allocations added later.
"""
function attach!(r::Residency, queue::Union{MTL.MTLCommandQueue,MTL.MTL4CommandQueue})
    set = r.set
    set === nothing || MTL.add_residency_set!(queue, set)
    return r
end

"""
    use_residency!(cb::MTL4CommandBuffer, r)

Declare `r`'s set on a Metal 4 command buffer, which the API asks for in addition to
the queue's. The set has to be complete before this: a command buffer that already
declared it does not pick up later allocations.
"""
function use_residency!(cb::MTL.MTL4CommandBuffer, r::Residency)
    set = r.set
    set === nothing || MTL.use_residency_set!(cb, set)
    return nothing
end

"""
    resident!(r, allocations) -> r
    resident!(r, allocation) -> r

Add each allocation (an `MTLBuffer`, an `MTLAccelerationStructure`, an
`MTLIndirectCommandBuffer`) that is not in the set yet, and commit the set once if
anything changed. `nothing` entries are skipped. A call that adds nothing is one
lookup per allocation, so it may be repeated for the same allocations.
"""
function resident!(r::Residency, allocations::Union{Tuple,AbstractVector})
    set = r.set
    set === nothing && return r
    Base.@lock r.lock begin
        changed = takedropped!(r, set)
        for a in allocations
            a === nothing && continue
            key = UInt(pointer(a))
            haskey(r.members, key) && continue
            gen = (r.generation += 1)
            r.members[key] = gen
            MTL.add_allocation!(set, a)
            # Membership ends with the wrapper. A wrapper without a finalizer (an
            # unmanaged one) stays until `forget!`.
            ismutable(a) && finalizer(x -> forget!(r, x, gen), a)
            changed = true
        end
        changed && MTL.commit!(set)
    end
    return r
end
resident!(r::Residency, allocation::Metal.ObjectiveC.Object) = resident!(r, (allocation,))

"""
    resident!(r, allocation, owner) -> r

Add `allocation`, which another library allocated and frees, until `owner` is
collected: the object whose collection means nothing reaches the allocation any
more. Membership then ends after a drain ([`forget_after_drain!`](@ref)), as a
replay in flight may still reach it. For an allocation whose wrapper has no
finalizer of its own (Metal.jl's `MTLBuffer`), which `resident!(r, allocation)`
would keep in the set for good.
"""
function resident!(r::Residency, allocation::Metal.ObjectiveC.Object, owner)
    r.set === nothing && return r
    key = UInt(pointer(allocation))
    known = Base.@lock r.lock haskey(r.members, key)
    known && return r
    resident!(r, (allocation,))
    # The membership this owner ends, and no later one at the same address.
    gen = Base.@lock r.lock get(r.members, key, 0)
    gen == 0 || finalizer(_ -> forget_after_drain!(r, allocation, gen), owner)
    return r
end

"""
    forget!(r, allocation)

Take `allocation` out of the set, so that releasing it frees it. For the owner that
frees an allocation now; the GPU must be done with it. Safe in a finalizer, which
applies the removal only if `lock` is free and otherwise leaves it to the next call
that takes `lock`.
"""
forget!(r::Residency, allocation) = forget!(r, allocation, 0)

function forget!(r::Residency, allocation, generation::Int)
    set = r.set
    set === nothing && return nothing
    while true
        old = @atomic r.dropped
        _, ok = @atomicreplace r.dropped old => Dropped(allocation, generation, old)
        ok && break
    end
    GC.in_finalizer() || return (trim!(r); nothing)
    # The task this finalizer interrupted may hold `lock` in the middle of changing
    # `members`, and a reentrant `trylock` would succeed there, so only a free lock.
    (!islocked(r.lock) && trylock(r.lock)) || return nothing
    try
        takedropped!(r, set) && MTL.commit!(set)
    finally
        unlock(r.lock)
    end
    return nothing
end

"""
    forget_after_drain!(r, allocation[, generation])

Take `allocation` out of the set once the work submitted before this call has finished,
which [`forget_drained!`](@ref) applies. For an allocation its owner freed while the GPU
may still reach it by address: a recorded command reads it through the set alone, and
removed now, a replay in flight would read zeros. Safe in a finalizer.

With `generation`, only that membership ends: one another owner ended meanwhile (the
pool's `forget!`) let the address go, and a later allocation there is not this one.
"""
function forget_after_drain!(r::Residency, allocation, generation::Int = 0)
    r.set === nothing && return nothing
    while true
        old = @atomic r.retired
        _, ok = @atomicreplace r.retired old => Retired(allocation, generation, old)
        ok && return nothing
    end
end

"""
    forget_drained!(r) -> r

Remove what [`forget_after_drain!`](@ref) queued. The caller has seen the device finish every
submission made before now (a drain, or a queue with nothing in flight).
"""
function forget_drained!(r::Residency)
    set = r.set
    set === nothing && return r
    node = @atomicswap r.retired = nothing
    node === nothing && return r
    Base.@lock r.lock begin
        changed = takedropped!(r, set)
        while node !== nothing
            key = UInt(pointer(node.allocation))
            gen = get(r.members, key, 0)
            if gen != 0 && (node.generation == 0 || node.generation == gen)
                delete!(r.members, key)
                MTL.remove_allocation!(set, node.allocation)
                changed = true
            end
            node = node.next
        end
        changed && MTL.commit!(set)
    end
    return r
end

"""
    trim!(r) -> r

Apply the removals finalizers left, and commit the set if any.
"""
function trim!(r::Residency)
    set = r.set
    set === nothing && return r
    Base.@lock r.lock begin
        takedropped!(r, set) && MTL.commit!(set)
    end
    return r
end

# Remove what `forget!` queued from `set`, `r`'s; `lock` held. Returns whether the
# set changed.
function takedropped!(r::Residency, set::MTL.MTLResidencySet)
    node = @atomicswap r.dropped = nothing
    changed = false
    while node !== nothing
        key = UInt(pointer(node.allocation))
        gen = get(r.members, key, 0)
        if gen != 0 && (node.generation == 0 || node.generation == gen)
            delete!(r.members, key)
            MTL.remove_allocation!(set, node.allocation)
            changed = true
        end
        node = node.next
    end
    return changed
end
