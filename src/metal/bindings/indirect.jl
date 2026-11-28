# Indirect command buffers: what a recorded plan replays.
#
# An `MTLCommandBuffer` is single-use: a committed command buffer cannot be committed
# again, so "record once, replay" cannot be built out of one. The replayable unit is
# the indirect command buffer: commands are encoded into it once, by the host or by a
# kernel, and a compute encoder replays a range of them with
# `executeCommandsInBuffer:withRange:` for as many frames as the plan lives. The
# types (descriptor, buffer, compute command, execution range) are in Metal.jl's
# generated `libmtl.jl`; these are the calls that use them.

using Metal.ObjectiveC: Foundation

"""
    indirect_command_buffer_descriptor(; command_types, max_kernel_buffers,
                                       inherit_pipeline, inherit_buffers, ray_tracing)

What an indirect command buffer may hold. A command may only set what the descriptor
declared: `max_kernel_buffers` bounds the buffer indices any command binds, and a
command binding past it does nothing, silently. `inherit_pipeline`/`inherit_buffers` make the
commands take the executing encoder's state instead of their own: false for a
recorded plan, whose executing encoder holds nothing. `ray_tracing` is required for a
command that traces rays; without it every ray misses, silently.
"""
function indirect_command_buffer_descriptor(;
        command_types::MTL.MTLIndirectCommandType = MTL.MTLIndirectCommandTypeConcurrentDispatch,
        max_kernel_buffers::Integer = 8, inherit_pipeline::Bool = false,
        inherit_buffers::Bool = false, ray_tracing::Bool = false)
    desc = @objc [MTLIndirectCommandBufferDescriptor new]::MTL.MTLIndirectCommandBufferDescriptor
    desc.commandTypes = command_types
    desc.maxKernelBufferBindCount = max_kernel_buffers
    desc.inheritPipelineState = inherit_pipeline
    desc.inheritBuffers = inherit_buffers
    desc.supportRayTracing = ray_tracing
    return desc
end

"""
    indirect_command_buffer(mtldevice, descriptor, maxcommands; storage) -> MTLIndirectCommandBuffer

Storage for `maxcommands` commands. `Private` by default; a buffer the host encodes
into needs `MTL.MTLResourceStorageModeShared` under API validation.
"""
function indirect_command_buffer(dev::MTL.MTLDevice, desc::MTL.MTLIndirectCommandBufferDescriptor,
                                 maxcommands::Integer;
                                 storage::MTL.MTLResourceOptions = MTL.MTLResourceStorageModePrivate)
    icb = @objc [dev::id{MTL.MTLDevice} newIndirectCommandBufferWithDescriptor:desc::id{MTL.MTLIndirectCommandBufferDescriptor}
                                         maxCommandCount:maxcommands::UInt64
                                         options:storage::MTL.MTLResourceOptions]::Union{Nothing,MTL.MTLIndirectCommandBuffer}
    icb === nothing && error("Metal refused an indirect command buffer of $maxcommands commands; " *
                             "check the descriptor's limits.")
    return icb
end

"""
    indirect_compute_command(icb, index) -> MTLIndirectComputeCommand

The command at one-based `index`, to encode into.
"""
indirect_compute_command(icb::MTL.MTLIndirectCommandBuffer, index::Integer) =
    @objc [icb::id{MTL.MTLIndirectCommandBuffer} indirectComputeCommandAtIndex:(index - 1)::UInt64]::MTL.MTLIndirectComputeCommand

"""
    set_pipeline!(cmd::MTLIndirectComputeCommand, pipeline)

The pipeline the command runs. It must have been built with
`supportIndirectCommandBuffers` (`Metal.mtlfunction(...; indirect = true)`); a command
refuses any other.
"""
set_pipeline!(cmd::MTL.MTLIndirectComputeCommand, pipeline::MTL.MTLComputePipelineState) =
    @objc [cmd::id{MTL.MTLIndirectComputeCommand} setComputePipelineState:pipeline::id{MTL.MTLComputePipelineState}]::Nothing

"""
    set_kernel_buffer!(cmd::MTLIndirectComputeCommand, buf, offset, index)

Bind `buf` at `offset` to the one-based buffer `index`. A command has no `setBytes`,
so arguments passed by value are packed into a buffer the command binds.
"""
set_kernel_buffer!(cmd::MTL.MTLIndirectComputeCommand, buf::MTL.MTLBuffer, offset::Integer,
                   index::Integer) =
    @objc [cmd::id{MTL.MTLIndirectComputeCommand} setKernelBuffer:buf::id{MTL.MTLBuffer}
                                                  offset:offset::UInt64
                                                  atIndex:(index - 1)::UInt64]::Nothing

"""
    dispatch_threadgroups!(cmd::MTLIndirectComputeCommand, groups, threads)

The command's grid, carried by the command rather than taken from the encoder.
"""
dispatch_threadgroups!(cmd::MTL.MTLIndirectComputeCommand, groups::MTL.MTLSize, threads::MTL.MTLSize) =
    @objc [cmd::id{MTL.MTLIndirectComputeCommand} concurrentDispatchThreadgroups:groups::MTL.MTLSize
                                                  threadsPerThreadgroup:threads::MTL.MTLSize]::Nothing

"""
    set_barrier!(cmd)

Make the command wait for every command before it in the buffer. Commands of a
`ConcurrentDispatch` buffer may otherwise run concurrently, so a pass boundary is a
barrier.
"""
set_barrier!(cmd::MTL.MTLIndirectComputeCommand) =
    @objc [cmd::id{MTL.MTLIndirectComputeCommand} setBarrier]::Nothing

"""
    execute_commands!(enc, icb, range)

Replay the commands of `icb` in the one-based `range`.
"""
function execute_commands!(enc::MTL.MTLComputeCommandEncoderLike, icb::MTL.MTLIndirectCommandBuffer,
                           range::UnitRange{<:Integer})
    r = Foundation.NSRange(first(range) - 1, length(range))
    @objc [enc::id{MTL.MTLComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTL.MTLIndirectCommandBuffer}
                                                 withRange:r::Foundation.NSRange]::Nothing
end

"""
    execute_commands!(enc, icb, rangebuf, offset)

Replay the range of commands the device wrote: `rangebuf` holds an
`MTLIndirectCommandBufferExecutionRange` at `offset`, two `UInt32`, a zero-based
location and a length, which the command processor reads when the encoder reaches this
point. A length of zero runs nothing. This is how a recorded plan keeps a `repeat!`
gate: a one-thread kernel writes the length the gate implies just before the range is
read, and the host never learns whether the iteration ran.
"""
function execute_commands!(enc::MTL.MTLComputeCommandEncoderLike, icb::MTL.MTLIndirectCommandBuffer,
                           rangebuf::MTL.MTLBuffer, offset::Integer)
    @objc [enc::id{MTL.MTLComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTL.MTLIndirectCommandBuffer}
                                                 indirectBuffer:rangebuf::id{MTL.MTLBuffer}
                                                 indirectBufferOffset:offset::UInt64]::Nothing
end
