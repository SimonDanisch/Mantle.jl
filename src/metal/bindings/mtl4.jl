# Metal 4 submission of a recorded plan: an unmanaged compute encoder for the
# per-submission path, dispatches on any Metal 4 compute encoder, argument tables with
# zeroed bindings, and a commit feedback handler that runs no Julia code that can
# allocate. The queue, allocator, command buffer, events and residency calls are
# Metal.jl's (`MTL.MTL4CommandQueue`, `MTL.beginCommandBuffer!`, `MTL.commit!`, ...).
#
# What differs from the legacy queue, and why it matters here. An `MTL4CommandBuffer`
# is reusable: `beginCommandBufferWithAllocator:` refills it from an
# `MTL4CommandAllocator`, a command pool whose `reset` is `vkResetCommandPool`.
# Completion is a timeline: there is no `waitUntilCompleted` and no completed handler
# on a command buffer; the queue signals an `MTLSharedEvent` with a value. The compute
# encoder has no `setBytes:` and no `setBuffer:` (arguments are GPU addresses in an
# `MTL4ArgumentTable`), no `useResource:` (residency is residency sets only) and no
# automatic hazard tracking (every dependency is an explicit barrier).
#
# Everything here needs macOS 26 or later.

using Metal.ObjectiveC: Foundation, @objcblock

@objcwrapper managed = false MTL4ComputeCommandEncoderRef <: MTL.MTL4ComputeCommandEncoder

@doc """
    MTL4ComputeCommandEncoderRef(cb::MTL4CommandBuffer)

A new compute encoder on `cb`, as an unmanaged wrapper whose one reference belongs to
the caller; `close` ends it and gives the reference back. The managed
`MTL.MTL4ComputeCommandEncoder(cb)` is a heap object per call, and a submission path
opens one per submission.
""" MTL4ComputeCommandEncoderRef

function MTL4ComputeCommandEncoderRef(cb::MTL.MTL4CommandBuffer)
    # `computeCommandEncoder` returns an autoreleased object. Retained inside a pool of
    # our own, which is drained right after: on a thread with no pool open the
    # autoreleased reference goes to one that is never drained, and the encoder leaks.
    pool = ccall(:objc_autoreleasePoolPush, Ptr{Cvoid}, ())
    try
        ptr = @objc [cb::id{MTL.MTL4CommandBuffer} computeCommandEncoder]::id{MTL4ComputeCommandEncoderRef}
        enc = MTL4ComputeCommandEncoderRef(ptr)
        Foundation.retain(enc)
        return enc
    finally
        ccall(:objc_autoreleasePoolPop, Cvoid, (Ptr{Cvoid},), pool)
    end
end

"""
    set_pipeline!(enc::MTL4ComputeCommandEncoder, pipeline)

Bind the compute pipeline the next dispatches run.
"""
set_pipeline!(enc::MTL.MTL4ComputeCommandEncoderLike, pipeline::MTL.MTLComputePipelineState) =
    @objc [enc::id{MTL.MTL4ComputeCommandEncoder} setComputePipelineState:pipeline::id{MTL.MTLComputePipelineState}]::Nothing

"""
    set_argument_table!(enc, table)

Bind the argument table the next dispatches read their buffer addresses from. The
bindings are captured when a dispatch is encoded.
"""
set_argument_table!(enc::MTL.MTL4ComputeCommandEncoderLike, table::MTL.MTL4ArgumentTable) =
    @objc [enc::id{MTL.MTL4ComputeCommandEncoder} setArgumentTable:table::id{MTL.MTL4ArgumentTable}]::Nothing

"""
    dispatch_threadgroups!(enc::MTL4ComputeCommandEncoder, groups, threads)

Dispatch `groups` threadgroups of `threads` threads.
"""
dispatch_threadgroups!(enc::MTL.MTL4ComputeCommandEncoderLike, groups::MTL.MTLSize, threads::MTL.MTLSize) =
    @objc [enc::id{MTL.MTL4ComputeCommandEncoder} dispatchThreadgroups:groups::MTL.MTLSize
                                                  threadsPerThreadgroup:threads::MTL.MTLSize]::Nothing

"""
    dispatch_threadgroups!(enc::MTL4ComputeCommandEncoder, buf, offset, threads)

Dispatch a grid the device decided: `buf` holds three `UInt32` threadgroup counts at
`offset`, and a zero runs nothing. This is how a `repeat!` gate survives without an
indirect command buffer: a replayed gate is a zero-length execution range, an encoded
one a zero-count grid, written by the same one-thread kernel, and the host never
learns whether the iteration ran. The selector takes a GPU address, so `buf` has to
be in a residency set the command buffer uses.
"""
function dispatch_threadgroups!(enc::MTL.MTL4ComputeCommandEncoderLike, buf::MTL.MTLBuffer,
                                offset::Integer, threads::MTL.MTLSize)
    address = UInt64(buf.gpuAddress) + UInt64(offset)
    @objc [enc::id{MTL.MTL4ComputeCommandEncoder} dispatchThreadgroupsWithIndirectBuffer:address::UInt64
                                                  threadsPerThreadgroup:threads::MTL.MTLSize]::Nothing
end

"""
    argument_table(mtldevice, buffers) -> MTL4ArgumentTable

An argument table with `buffers` buffer bindings, all zeroed at creation
(`initializeBindings`). Without it a binding that is never written is undefined
rather than null.
"""
function argument_table(dev::MTL.MTLDevice, buffers::Integer)
    desc = MTL.MTL4ArgumentTableDescriptor()
    desc.maxBufferBindCount = buffers
    desc.initializeBindings = true
    return MTL.MTL4ArgumentTable(dev, desc)
end

# When a submission fails. Metal 4 has no `waitUntilCompleted` and no error property on
# a command buffer, so a submission that faults reports nothing: the event it would
# have signalled never advances, and a caller waiting on that value blocks forever
# with nothing in the system log. That is how a GPU fault in a replayed plan presents.
#
# `commit:count:options:` with a feedback handler is the error path. The handler runs
# on a dispatch worker, so it does the minimum that is safe there: read the error out
# of the feedback and keep it. Reading the message is the waiter's job, on a Julia
# thread. Metal.jl's `add_feedback_handler!` wraps the feedback in a managed object
# and catches errors, which allocates on that thread; this handler does not.

"""
    MTL4Feedback()

Commit options whose feedback handler keeps the first error a submission reports, for
[`failure`](@ref), for [`commit!`](@ref).
"""
mutable struct MTL4Feedback
    options::MTL.MTL4CommitOptions
    # The retained `id<NSError>` of the first failure, as an integer so the handler
    # stores a word and touches nothing that can allocate or yield.
    err::Base.RefValue{UInt}
    # Rooted: the block must outlive every submission using it.
    block::Any
end

function MTL4Feedback()
    options = MTL.MTL4CommitOptions()
    err = Ref(UInt(0))
    # No allocation, no lock, no yield: two message sends and one store. The error is
    # retained because the feedback, and what it holds, is only valid in the handler.
    handler = function (feedback::Ptr{Cvoid})
        e = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), feedback,
                  ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "error"))
        if e != C_NULL && err[] == 0
            err[] = UInt(ccall(:objc_retain, Ptr{Cvoid}, (Ptr{Cvoid},), e))
        end
        return nothing
    end
    block = @objcblock(handler, Nothing, (Ptr{Cvoid},))
    @objc [options::id{MTL.MTL4CommitOptions} addFeedbackHandler:block::id{Foundation.NSBlock}]::Nothing
    return MTL4Feedback(options, err, block)
end

"""
    commit!(queue::MTL4CommandQueue, cbs::Vector{id{MTL4CommandBuffer}}, feedback)

Submit the command buffers `cbs` points to with `feedback`'s options, so a failure is
reportable. Over pointers the caller converted once and keeps, so a submission
allocates nothing; Metal.jl's `commit!` builds the pointer array on every call. The
caller keeps the command buffers alive.
"""
function commit!(q::MTL.MTL4CommandQueue, cbs::Vector{id{MTL.MTL4CommandBuffer}}, fb::MTL4Feedback)
    isempty(cbs) && return nothing
    @objc [q::id{MTL.MTL4CommandQueue} commit:cbs::Ptr{id{MTL.MTL4CommandBuffer}}
                                       count:length(cbs)::UInt64
                                       options:fb.options::id{MTL.MTL4CommitOptions}]::Nothing
end

"""
    failure(feedback) -> String or nothing

The message of the first submission that failed with these options, if any.
"""
function failure(fb::MTL4Feedback)
    e = fb.err[]
    e == 0 && return nothing
    return String(Foundation.NSError(id{Foundation.NSError}(e)).localizedDescription)
end
