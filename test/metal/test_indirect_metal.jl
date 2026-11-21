# Recording a plan once and replaying it, which on Metal means an indirect command
# buffer, through Mantle's bindings (`src/metal/bindings/indirect.jl`).
#
# An `MTLCommandBuffer` is single-use: it cannot be committed twice, so the way a
# Vulkan backend bakes a render graph (record a `VkCommandBuffer` once, re-submit it
# every frame) has no counterpart built out of one. `MTLIndirectCommandBuffer` is the
# counterpart: commands are encoded into it once and a compute encoder replays the
# range with `executeCommandsInBuffer:withRange:` for as long as the plan lives.
#
# Three things have to hold for a graph to be recorded onto it, and each is a test:
#
#   1. a kernel compiled by Metal.jl can be encoded into a command and replayed;
#   2. its arguments can come from a buffer, because a command may only
#      `setKernelBuffer`: there is no `setBytes` on an indirect command, so the
#      arguments are packed once into memory the command points at;
#   3. commands are concurrent unless told otherwise, so a pass boundary needs
#      `setBarrier`.
module IndirectCommandBufferMetal

using Test, Mantle, Metal
const MTL = Metal.MTL
const MB = Mantle.MetalBindings

function icb_add_const!(a, v::Float32)
    i = Metal.thread_position_in_grid_1d()
    @inbounds a[i] += v
    return nothing
end

# A pipeline the commands of an indirect command buffer can use: they refuse one built
# without `supportIndirectCommandBuffers`, which `indirect = true` sets.
icbkernel(f, tt, name) = Metal.mtlfunction(f, tt; name, indirect = true)

# An array's buffer, which `useResource:` takes.
bufferof(a::Metal.MtlArray) = a.data[].buffer

"""Write `x`'s bytes at `base + off`."""
function pack!(base::Ptr{UInt8}, x, off::Integer)
    r = Ref(x)
    GC.@preserve r unsafe_copyto!(base + off, Ptr{UInt8}(Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
    return nothing
end

"""
The argument block a command binds, packed once: the kernel state at index 1, then
one slot per argument, which is what a launch pushes per call with `setBytes:`. A
buffer bound at index `i` and bytes pushed at index `i` are the same binding to the
shader, which is what lets a recorded command carry arguments at all. 256-byte slots,
since a buffer offset has an alignment rule; a plan packs tightly against `argsize`.
"""
function icb_pack(kernel, args...)
    argbuf = MTL.MTLBuffer(Metal.device(), 256 * (1 + length(args)); storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTL.contents(argbuf))
    pack!(base, Metal.KernelState(kernel; seed = UInt32(1)), 0)
    for (k, a) in enumerate(args)
        pack!(base, a, 256 * k)
    end
    return argbuf
end

"""Bind the kernel state and `nargs` arguments of `argbuf` to command `c`."""
function bindargs!(c, argbuf, nargs)
    MB.set_kernel_buffer!(c, argbuf, 0, 1)
    for k in 1:nargs
        MB.set_kernel_buffer!(c, argbuf, 256 * k, k + 1)
    end
    return c
end

"""Encode `n` commands, all the same dispatch, optionally serialised."""
function icb_encode(kernel, argbuf, nargs, n; barrier::Bool)
    desc = MB.indirect_command_buffer_descriptor(; max_kernel_buffers = nargs + 1)
    icb = MB.indirect_command_buffer(Metal.device(), desc, n; storage = MTL.MTLResourceStorageModeShared)
    for i in 1:n
        c = bindargs!(MB.indirect_compute_command(icb, i), argbuf, nargs)
        MB.set_pipeline!(c, kernel.pipeline)
        MB.dispatch_threadgroups!(c, MTL.MTLSize(1), MTL.MTLSize(64))
        barrier && MB.set_barrier!(c)
    end
    return icb
end

"""Encode, commit and wait for one command buffer, with `resources` declared."""
function onebuffer(f, queue, resources)
    cb = MTL.MTLCommandBuffer(queue)
    enc = MTL.MTLComputeCommandEncoder(cb)
    # The commands reach their buffers by address, and an address the encoder was
    # never told about is not resident: it reads as zeros rather than faulting.
    for r in resources
        MTL.use!(enc, r, MTL.ReadWriteUsage)
    end
    f(enc)
    close(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    @test cb.status == MTL.MTLCommandBufferStatusCompleted
    return nothing
end

"""One replay: the whole of a recorded frame's host work."""
icb_replay!(queue, icb, range, resources) =
    onebuffer(enc -> MB.execute_commands!(enc, icb, range), queue, resources)

zeroed(T, n) = (a = Metal.MtlVector{T}(undef, n); fill!(a, zero(T)); Metal.synchronize(); a)

const ADD_TT = Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}

@testset "a compiled kernel replays out of an indirect command buffer" begin
    kernel = icbkernel(icb_add_const!, ADD_TT, "icb_add")
    out = zeroed(Float32, 64)
    argbuf = icb_pack(kernel, Metal.mtlconvert(out), 2.5f0)
    icb = icb_encode(kernel, argbuf, 2, 1; barrier = false)
    queue = MTL.MTLCommandQueue(Metal.device())
    res = (argbuf, bufferof(out), Metal.kernel_resources(kernel)...)

    # Encoded once, above. Each replay runs it again with nothing re-encoded, which
    # is the property the whole recording path rests on.
    icb_replay!(queue, icb, 1:1, res)
    @test all(==(2.5f0), Array(out))
    icb_replay!(queue, icb, 1:1, res)
    @test all(==(5.0f0), Array(out))
    icb_replay!(queue, icb, 1:1, res)
    @test all(==(7.5f0), Array(out))
end

@testset "commands in one buffer are concurrent until a barrier says otherwise" begin
    kernel = icbkernel(icb_add_const!, ADD_TT, "icb_add2")
    queue = MTL.MTLCommandQueue(Metal.device())
    n = 32

    # With barriers: 32 read-modify-writes of the same memory, one after another, so
    # the count is exact. This is what a pass boundary has to encode.
    out = zeroed(Float32, 64)
    argbuf = icb_pack(kernel, Metal.mtlconvert(out), 1f0)
    icb = icb_encode(kernel, argbuf, 2, n; barrier = true)
    icb_replay!(queue, icb, 1:n, (argbuf, bufferof(out), Metal.kernel_resources(kernel)...))
    @test all(==(Float32(n)), Array(out))

    # Without: the same 32 commands race, and the sum comes out short. Asserted
    # because it is the reason `set_barrier!` exists: a plan whose passes are recorded
    # without one would lose writes exactly like this, and silently.
    out2 = zeroed(Float32, 64)
    argbuf2 = icb_pack(kernel, Metal.mtlconvert(out2), 1f0)
    icb2 = icb_encode(kernel, argbuf2, 2, n; barrier = false)
    icb_replay!(queue, icb2, 1:n, (argbuf2, bufferof(out2), Metal.kernel_resources(kernel)...))
    @test maximum(Array(out2)) < Float32(n)
end

@testset "two replays on one encoder run in order" begin
    kernel = icbkernel(icb_add_const!, ADD_TT, "icb_add3")
    queue = MTL.MTLCommandQueue(Metal.device())
    n = 32

    # The same 32 read-modify-writes as the concurrency test, but as 32 separate
    # `executeCommandsInBuffer` calls with no barrier encoded in any command. A
    # compute encoder is serial unless it was made concurrent, and a replay is one
    # command in it, so the sum is exact where a single replay of the same 32 raced
    # and came out short. This is what a recorded plan's pass boundaries rest on:
    # separate replays need no barrier of their own.
    out = zeroed(Float32, 64)
    argbuf = icb_pack(kernel, Metal.mtlconvert(out), 1f0)
    icb = icb_encode(kernel, argbuf, 2, n; barrier = false)
    onebuffer(queue, (argbuf, bufferof(out), Metal.kernel_resources(kernel)...)) do enc
        for i in 1:n
            MB.execute_commands!(enc, icb, i:i)
        end
    end
    @test all(==(Float32(n)), Array(out))
end

# The range writer: one thread, deciding at execution time how much of the buffer the
# next replay runs. `flag` is what a `repeat!` gate writes; `len` is how many commands
# the iteration holds.
function icb_write_range!(range, flag, loc::UInt32, len::UInt32)
    @inbounds range[1] = loc
    @inbounds range[2] = flag[1] != 0 ? len : UInt32(0)
    return nothing
end

@testset "a device-written execution range decides what a replay runs" begin
    dev = Metal.device()
    queue = MTL.MTLCommandQueue(dev)
    add = icbkernel(icb_add_const!, ADD_TT, "icb_add4")
    writer = icbkernel(icb_write_range!,
                       Tuple{Metal.MtlDeviceVector{UInt32,1}, Metal.MtlDeviceVector{UInt32,1}, UInt32, UInt32},
                       "icb_range")

    out = Metal.MtlVector{Float32}(undef, 64)
    # Two elements each, not one: a four-byte buffer an indirect command reaches by
    # address is not made resident by `useResource` (the last testset here).
    range = Metal.MtlVector{UInt32}(undef, 2)
    flag = Metal.MtlVector{UInt32}(undef, 2)
    ngated = 4

    # One buffer, two kinds of command: slot 1 writes the execution range, slots 2:5
    # are the gated work. The two kernels' states sit in their own slots, encoded
    # once for both halves of the test.
    argbuf = MTL.MTLBuffer(dev, 256 * 8; storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTL.contents(argbuf))
    pack!(base, Metal.KernelState(writer; seed = UInt32(1)), 0)
    pack!(base, Metal.mtlconvert(range), 256)      # the range writer's arguments
    pack!(base, Metal.mtlconvert(flag), 512)
    pack!(base, UInt32(1), 768)                    # location: command 2, zero-based
    pack!(base, UInt32(ngated), 1024)              # length, if the gate is open
    pack!(base, Metal.KernelState(add; seed = UInt32(1)), 1280)
    pack!(base, Metal.mtlconvert(out), 1536)       # the gated work's arguments
    pack!(base, 1f0, 1792)

    desc = MB.indirect_command_buffer_descriptor(; max_kernel_buffers = 5)
    icb = MB.indirect_command_buffer(dev, desc, 1 + ngated; storage = MTL.MTLResourceStorageModeShared)
    c = bindargs!(MB.indirect_compute_command(icb, 1), argbuf, 4)
    MB.set_pipeline!(c, writer.pipeline)
    MB.dispatch_threadgroups!(c, MTL.MTLSize(1), MTL.MTLSize(1))
    for i in 1:ngated
        g = MB.indirect_compute_command(icb, 1 + i)
        MB.set_kernel_buffer!(g, argbuf, 1280, 1)
        MB.set_kernel_buffer!(g, argbuf, 1536, 2)
        MB.set_kernel_buffer!(g, argbuf, 1792, 3)
        MB.set_pipeline!(g, add.pipeline)
        MB.dispatch_threadgroups!(g, MTL.MTLSize(1), MTL.MTLSize(64))
        MB.set_barrier!(g)
    end

    resources = (argbuf, bufferof(out), bufferof(range), bufferof(flag),
                 Metal.kernel_resources(writer)..., Metal.kernel_resources(add)...)
    gated_replay!() = onebuffer(queue, resources) do enc
        MB.execute_commands!(enc, icb, 1:1)                               # write the range
        MB.execute_commands!(enc, icb, bufferof(range), range.offset)     # run what it says
    end

    # Gate closed: the same replay runs nothing, and the host said nothing about it:
    # the length the command processor read was written by a kernel one command
    # earlier.
    fill!(out, 0f0); fill!(flag, UInt32(0)); Metal.synchronize()
    gated_replay!()
    @test all(==(0f0), Array(out))
    @test Array(range) == UInt32[1, 0]

    # Gate open: four commands, serialised by their own barriers.
    fill!(flag, UInt32(1)); Metal.synchronize()
    gated_replay!()
    @test all(==(Float32(ngated)), Array(out))
    @test Array(range) == UInt32[1, ngated]
end

# A four-byte buffer an indirect command reaches by address is not made resident.
#
# `useResource` covers it for an ordinary dispatch, where the argument is packed bytes
# holding the same raw address, and does not cover it for a command replayed out of
# an indirect buffer: the write below is dropped and a read of the same buffer comes
# back as zero, silently, as an unmapped page does. Eight bytes is enough; four is
# not. Pinned rather than worked around because the failure has no symptom other than
# wrong numbers, and because a driver that fixes it should make this test say so.
#
# Mantle's pool never hands out one (its blocks are megabytes and a `GPURef` is a
# slice of one), so this is a hazard for hand-built buffers, which `record!` checks
# for (`checkresident`).
function icb_store!(a, v::UInt32)
    @inbounds a[1] = v
    return nothing
end

@testset "a four-byte resource is not resident for an indirect command" begin
    dev = Metal.device()
    queue = MTL.MTLCommandQueue(dev)
    kernel = icbkernel(icb_store!, Tuple{Metal.MtlDeviceVector{UInt32,1}, UInt32}, "icb_store")

    function store_through_icb(n)
        a = Metal.MtlVector{UInt32}(undef, n); fill!(a, UInt32(9)); Metal.synchronize()
        argbuf = icb_pack(kernel, Metal.mtlconvert(a), UInt32(42))
        desc = MB.indirect_command_buffer_descriptor(; max_kernel_buffers = 3)
        icb = MB.indirect_command_buffer(dev, desc, 1; storage = MTL.MTLResourceStorageModeShared)
        c = bindargs!(MB.indirect_compute_command(icb, 1), argbuf, 2)
        MB.set_pipeline!(c, kernel.pipeline)
        MB.dispatch_threadgroups!(c, MTL.MTLSize(1), MTL.MTLSize(1))
        icb_replay!(queue, icb, 1:1, (argbuf, bufferof(a), Metal.kernel_resources(kernel)...))
        return Array(a)[1]
    end

    @test store_through_icb(1) == UInt32(9)      # dropped: four bytes
    @test store_through_icb(2) == UInt32(42)     # landed: eight

    # The same four-byte buffer, from an ordinary launch, takes the store: the buffer
    # is fine and the indirect path is what loses it.
    a = Metal.MtlVector{UInt32}(undef, 1); fill!(a, UInt32(9)); Metal.synchronize()
    Metal.@metal threads=1 groups=1 icb_store!(a, UInt32(42))
    Metal.synchronize()
    @test Array(a)[1] == UInt32(42)
end

end
