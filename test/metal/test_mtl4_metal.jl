# Metal 4, which Mantle's `MTL4Queue` submits recorded plans with
# (`src/metal/bindings/mtl4.jl`), and what it asks of a caller.
#
# What the runtime says the model is: `MTL4ComputeCommandEncoder` has no `setBytes:`
# and no `setBuffer:` (arguments are GPU addresses in an `MTL4ArgumentTable`), no
# `useResource:` (residency is residency sets only), and no automatic hazard
# tracking, so every dependency the driver infers on a legacy queue becomes an
# explicit barrier. A recorded command already binds buffer slices by address, which
# is the MTL4 model.
#
# It is not faster. Measured 2026-09-10 on an M5, interleaved (six batches of 150
# submits, each batch's minimum, because a single before/after pass drifts by more
# than the effect): a one-segment ICB replay is 4.46 us on the legacy queue against
# 5.08 us on MTL4, and a one-kernel launch 5.29 against 5.92. A first,
# non-interleaved pass read the opposite and looked like a 2.3 us win for MTL4. Do
# not quote a Metal host-time comparison that was not interleaved.
module MTL4Metal

using Test, Mantle, Metal, KernelAbstractions
const MTL = Metal.MTL
const MB = Mantle.MetalBindings
const Foundation = Metal.ObjectiveC.Foundation

const MTL4_SELS = Dict(
    "MTL4CommandQueue"          => ["commit:count:", "commit:count:options:", "addResidencySet:",
                                    "signalEvent:value:", "waitForEvent:value:"],
    "MTL4CommandAllocator"      => ["reset"],
    "MTL4CommandBuffer"         => ["beginCommandBufferWithAllocator:", "computeCommandEncoder",
                                    "endCommandBuffer", "useResidencySet:"],
    "MTL4ComputeCommandEncoder" => ["setArgumentTable:", "setComputePipelineState:",
                                    "dispatchThreadgroups:threadsPerThreadgroup:",
                                    "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:",
                                    "executeCommandsInBuffer:withRange:",
                                    "executeCommandsInBuffer:indirectBuffer:"],
    "MTL4ArgumentTable"         => ["setAddress:atIndex:", "setResource:atBufferIndex:"])

struct ObjCMethodDesc
    name::Ptr{Cvoid}
    types::Cstring
end

"""Every selector a protocol declares, required or not, instance or class."""
function protocol_selectors(name::AbstractString)
    p = ccall(:objc_getProtocol, Ptr{Cvoid}, (Cstring,), name)
    p == C_NULL && return nothing
    out = String[]
    for req in (true, false), inst in (true, false)
        n = Ref{Cuint}(0)
        list = ccall(:protocol_copyMethodDescriptionList, Ptr{ObjCMethodDesc},
                     (Ptr{Cvoid}, Bool, Bool, Ptr{Cuint}), p, req, inst, n)
        list == C_NULL && continue
        for i in 1:n[]
            push!(out, unsafe_string(ccall(:sel_getName, Cstring, (Ptr{Cvoid},),
                                           unsafe_load(list, i).name)))
        end
        Libc.free(list)
    end
    return unique(out)
end

sel(name) = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), name)
rawptr(x) = reinterpret(Ptr{Cvoid}, pointer(x))
msg0(obj, s) = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), obj, sel(s))
supports(dev, s) = ccall(:objc_msgSend, Bool, (Ptr{Cvoid}, Ptr{Cvoid}), rawptr(dev), sel(s))
release(obj) = ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), obj, sel("release"))

function mtl4_add!(a, v::Float32)
    i = Metal.thread_position_in_grid_1d()
    @inbounds a[i] += v
    return nothing
end

const ADD_TT = Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}

@kernel function mtl4_plan_add!(out)
    i = @index(Global)
    @inbounds out[i] += 1f0
end

"""Write `x`'s bytes at `base + off`."""
function pack!(base::Ptr{UInt8}, x, off::Integer)
    r = Ref(x)
    GC.@preserve r unsafe_copyto!(base + off, Ptr{UInt8}(Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
    return nothing
end

"""
The arguments of `mtl4_add!` in a buffer, because MTL4 has no `setBytes`: the kernel
state at slot 1, then one 256-byte slot per argument, the layout a launch pushes.
"""
function packargs(kernel, a, value)
    argbuf = MTL.MTLBuffer(Metal.device(), 1024; storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTL.contents(argbuf))
    pack!(base, Metal.KernelState(kernel; seed = UInt32(1)), 0)
    pack!(base, Metal.mtlconvert(a), 256)
    pack!(base, value, 512)
    return argbuf
end

"""
One dispatch of `mtl4_add!` submitted by raw `objc_msgSend`, through an argument table
or replayed out of an indirect command buffer (`useicb`). Raw on purpose: it says
which selectors exist and in what order they have to be called, independently of the
wrappers `mtl4_indirect_probe!` uses.
"""
function mtl4_probe!(a::Metal.MtlVector{Float32}, value::Float32, useicb::Bool)
    dev = Metal.device()
    d = rawptr(dev)
    kernel = Metal.mtlfunction(mtl4_add!, ADD_TT; name = "mtl4_add", indirect = useicb)
    argbuf = packargs(kernel, a, value)

    # Residency is residency sets only: everything the dispatch reaches by address.
    residency = MB.Residency(dev)
    MB.resident!(residency, (argbuf, a.data[].buffer, Metal.kernel_resources(kernel)...))

    icb = nothing
    if useicb
        desc = MB.indirect_command_buffer_descriptor(; max_kernel_buffers = 3, ray_tracing = true)
        icb = MB.indirect_command_buffer(dev, desc, 1; storage = MTL.MTLResourceStorageModeShared)
        c = MB.indirect_compute_command(icb, 1)
        MB.set_pipeline!(c, kernel.pipeline)
        for (i, off) in enumerate((0, 256, 512))
            MB.set_kernel_buffer!(c, argbuf, off, i)
        end
        MB.dispatch_threadgroups!(c, MTL.MTLSize(1), MTL.MTLSize(64))
        MB.resident!(residency, icb)
    end

    q4 = msg0(d, "newMTL4CommandQueue")
    allocator = msg0(d, "newCommandAllocator")
    cb4 = msg0(d, "newCommandBuffer")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          q4, sel("addResidencySet:"), rawptr(residency.set))

    # The argument table replaces every `setBuffer`/`setBytes` a launch would do.
    atd = msg0(msg0(ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "MTL4ArgumentTableDescriptor"),
                    "alloc"), "init")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, UInt64), atd, sel("setMaxBufferBindCount:"), 4)
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Bool), atd, sel("setInitializeBindings:"), true)
    err = Ref{Ptr{Cvoid}}(C_NULL)
    table = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
                  d, sel("newArgumentTableWithDescriptor:error:"), atd, err)
    @test table != C_NULL
    gpu = UInt64(argbuf.gpuAddress)
    for (i, off) in enumerate((0, 256, 512))
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, UInt64, Csize_t),
              table, sel("setAddress:atIndex:"), gpu + UInt64(off), i - 1)
    end

    # One frame. The command buffer and the allocator are reusable, which is the one
    # thing MTL4 gives that the legacy queue cannot.
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), allocator, sel("reset"))
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          cb4, sel("beginCommandBufferWithAllocator:"), allocator)
    enc = msg0(cb4, "computeCommandEncoder")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          cb4, sel("useResidencySet:"), rawptr(residency.set))
    if useicb
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Foundation.NSRange),
              enc, sel("executeCommandsInBuffer:withRange:"), rawptr(icb), Foundation.NSRange(0, 1))
    else
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), enc, sel("setArgumentTable:"), table)
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
              enc, sel("setComputePipelineState:"), rawptr(kernel.pipeline))
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, MTL.MTLSize, MTL.MTLSize),
              enc, sel("dispatchThreadgroups:threadsPerThreadgroup:"), MTL.MTLSize(1), MTL.MTLSize(64))
    end
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), enc, sel("endEncoding"))
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cb4, sel("endCommandBuffer"))
    cbs = Ref(cb4)
    GC.@preserve cbs ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t),
                           q4, sel("commit:count:"), Base.unsafe_convert(Ptr{Ptr{Cvoid}}, cbs), 1)

    # Completion is an event: there is no `waitUntilCompleted` on an MTL4 buffer.
    ev = MTL.MTLSharedEvent(dev)
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt64),
          q4, sel("signalEvent:value:"), rawptr(ev), UInt64(1))
    while ev.signaledValue < 1
        yield()
    end
    foreach(release, (table, atd, cb4, allocator, q4))
    GC.@preserve kernel argbuf icb residency nothing
    return nothing
end

"""
The same dispatch with its grid read from memory, through Mantle's bindings: an
unmanaged encoder, an argument table with zeroed bindings, the pointer-array commit
with a feedback handler.

`dispatchThreadgroupsWithIndirectBuffer:` is the only way an MTL4 encoder runs a count
that was not known when the command was encoded, and a count of zero has to run
nothing. Both halves matter to a gate built out of it: a driver that clamped zero to
one would turn "this iteration is discarded" into "this iteration runs once more",
silently.
"""
function mtl4_indirect_probe!(a::Metal.MtlVector{Float32}, value::Float32, groups::Integer)
    dev = Metal.device()
    kernel = Metal.mtlfunction(mtl4_add!, ADD_TT; name = "mtl4_add_indirectgrid")
    argbuf = packargs(kernel, a, value)

    # Three `UInt32`, which is what the command processor fetches: a threadgroup count
    # per dimension. `groups = 0` is the closed gate.
    gridbuf = MTL.MTLBuffer(dev, 16; storage = Metal.SharedStorage)
    let p = convert(Ptr{UInt32}, MTL.contents(gridbuf))
        unsafe_store!(p, UInt32(groups), 1)
        unsafe_store!(p, UInt32(1), 2)
        unsafe_store!(p, UInt32(1), 3)
    end

    q = MTL.MTL4CommandQueue(dev)
    residency = MB.Residency(dev)
    MB.resident!(residency, (argbuf, gridbuf, a.data[].buffer, Metal.kernel_resources(kernel)...))
    MB.attach!(residency, q)
    alloc = MTL.MTL4CommandAllocator(dev)
    cb = MTL.MTL4CommandBuffer(dev)

    table = MB.argument_table(dev, 4)
    let gpu = UInt64(argbuf.gpuAddress)
        for (i, off) in enumerate((0, 256, 512))
            MTL.set_address!(table, gpu + UInt64(off), i)
        end
    end

    MTL.beginCommandBuffer!(cb, alloc)
    enc = MB.MTL4ComputeCommandEncoderRef(cb)
    MB.use_residency!(cb, residency)
    MB.set_argument_table!(enc, table)
    MB.set_pipeline!(enc, kernel.pipeline)
    MB.dispatch_threadgroups!(enc, gridbuf, 0, MTL.MTLSize(64))
    close(enc)
    MTL.endCommandBuffer!(cb)
    feedback = MB.MTL4Feedback()
    MB.commit!(q, [pointer(cb)], feedback)

    ev = MTL.MTLSharedEvent(dev)
    MTL.signal_event!(q, ev, UInt64(1))
    while ev.signaledValue < 1
        yield()
    end
    @test MB.failure(feedback) === nothing
    GC.@preserve kernel argbuf gridbuf table alloc cb residency nothing
    return nothing
end

zeroed(n) = (a = Metal.MtlVector{Float32}(undef, n); fill!(a, 0f0); Metal.synchronize(); a)

@testset "Metal 4" begin
    dev = Metal.device()
    # The selector is new in macOS 26; an older system raises on it.
    if Metal.macos_version() < v"26" || !supports(dev, "supportsMTL4CommandQueue")
        @test_skip "this device has no Metal 4 command queue"
    else
        @testset "the model a submission has to adopt" begin
            for (proto, wanted) in MTL4_SELS
                sels = protocol_selectors(proto)
                @test sels !== nothing
                for w in wanted
                    @test w in sels
                end
            end
            enc = protocol_selectors("MTL4ComputeCommandEncoder")
            # The three absences: arguments by address, residency by set, hazards by hand.
            @test !any(s -> startswith(s, "setBytes"), enc)
            @test !any(s -> startswith(s, "setBuffer"), enc)
            @test !any(s -> startswith(s, "useResource"), enc)
        end

        @testset "a kernel runs through an MTL4 argument table" begin
            a = zeroed(64)
            mtl4_probe!(a, 2.5f0, false)
            @test all(==(2.5f0), Array(a))
        end

        # A single short replay completes; `executeCommandsInBuffer:` on an MTL4
        # encoder stops completing after about a second of cumulative replayed work
        # (M5), which is why `MTL4Queue` encodes recordings instead (`canreplay`).
        @testset "an indirect command buffer replays on an MTL4 queue" begin
            a = zeroed(64)
            mtl4_probe!(a, 1.5f0, true)
            @test all(==(1.5f0), Array(a))
        end

        @testset "a threadgroup count read from memory decides the grid" begin
            a = zeroed(64)
            mtl4_indirect_probe!(a, 3.5f0, 1)
            @test all(==(3.5f0), Array(a))
            # And zero runs nothing, which is the half a gate is built on.
            mtl4_indirect_probe!(a, 3.5f0, 0)
            @test all(==(3.5f0), Array(a))
        end

        @testset "a recorded plan runs on a Mantle MTL4 queue" begin
            d = Mantle.MetalDevice(dev, Mantle.MTL4Queue(dev))
            out = Mantle.Buffer(d, zeros(Float32, 64))
            g = Mantle.Graph(d)
            Mantle.dispatch!(g, mtl4_plan_add!, (out,), 64; name = "mtl4 add")
            pl = Mantle.record!(Mantle.Plan(g; budget = Inf))
            try
                for _ in 1:3
                    Mantle.run!(pl)
                end
                Mantle.waitfor!(pl)
                @test Array(Mantle.storage(out)) == fill(3f0, 64)
                @test MB.failure(d.queue.feedback) === nothing
            finally
                Mantle.free!(pl)
            end
        end
    end
end

end
