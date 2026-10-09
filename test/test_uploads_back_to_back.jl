"""
Work handed to the device back to back, with nothing waited on in between, all
lands, in order.

Every path that reaches the device — an ad hoc launch, an upload, a readback, a
plan's run with a store in front of it — closes what it wrote inside the call
and hands it over. Two uploads in a row are the sharp case: the staging bytes
belong to the submission that copies out of them, so a second upload must never
reuse a buffer the first is still reading. A queue-level staging buffer could not
promise that once every transfer became its own submission.

What a user sees is the numbers, and that is what is asserted. How many
submissions and head barriers each call makes, and that the queue has nowhere to
keep an open command buffer, are asserted from the Vulkan queue's own counters in
`test/vulkan/test_closed_command_buffers.jl`, which this file was split from: no
portable submission count exists.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function ccb_bump!(a)
    i = @index(Global)
    @inbounds a[i] += 1f0
end

@kernel function ccb_add!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

@testset "every kind of call, back to back" begin
    dev = Mantle.Device(TESTBACKEND)
    be = Mantle.backend(dev)
    n = 256

    # An ad hoc launch…
    a = KA.allocate(be, Float32, n)
    fill!(a, 0f0)
    ccb_bump!(be, 64)(a; ndrange = n)
    # …an upload and a readback…
    b = Mantle.Buffer(dev, zeros(Float32, n))
    Mantle.update!(b, fill(3f0, n))
    @test Array(b) == fill(3f0, n)
    @test Array(a) == fill(1f0, n)

    # …and a recorded run, alone and with a store pending in front of it.
    kref = Mantle.GPURef(dev, Int32(2))
    out = Mantle.Buffer(dev, zeros(Int32, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, ccb_add!, (out, kref), n; name = "add")
    # `budget = Inf`: recorded whole, never measured; how a plan is cut by its
    # budget is `test_partitioned_recording.jl`.
    pl = Base.invokelatest(Mantle.Plan, g; budget = Inf)
    Mantle.record!(pl)
    Mantle.run!(pl)
    kref[] = Int32(5)
    Mantle.run!(pl)
    Mantle.waitfor!(pl)
    @test Array(out) == fill(Int32(7), n)
    Mantle.free!(pl)
end

@testset "two uploads back to back are both correct with nothing waited on between" begin
    dev = Mantle.Device(TESTBACKEND)
    n = 1 << 16
    x, y = rand(Float32, n), rand(Float32, n)
    bx = Mantle.Buffer(dev, zeros(Float32, n))
    by = Mantle.Buffer(dev, zeros(Float32, n))
    Mantle.update!(bx, x)
    Mantle.update!(by, y)
    @test Array(bx) == x
    @test Array(by) == y
    for _ in 1:8
        x .= rand(Float32, n); y .= rand(Float32, n)
        Mantle.update!(bx, x)
        Mantle.update!(by, y)
    end
    @test Array(bx) == x
    @test Array(by) == y
end
