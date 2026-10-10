"""
A buffer that outlives a device reset must not call into the device it outlived.

    d = KA.allocate(backend, Float32, 1000); fill!(d, 1f0)
    reset_device!(dev); d = nothing; GC.gc()      # <- SIGSEGV

Ten lines, and it took down the whole Vulkan suite. The reset dropped the old
context, which made it garbage — and its buffers garbage in the **same
collection**, where Julia does not order finalizers. Run the context's first and
the driver's device is destroyed; the buffer's finalizer then asked its timeline
semaphore a question and the driver took the fault. The finalizer did gate on
"device lost", and the reset's comment cited that gate as the reason this was
safe; the gate was simply never true for a voluntary reset. So a reset RETIRES
the device it replaces, and what this pins is that a retired device stays retired
and the one that replaces it is new and works.

It reproduced long before the per-device work, and the AMD laptop hit the same
thing from three call sites and filed it as "a floating GC race". It floats
because the crash lands wherever the next GC does.

**This crashes the process when it regresses**, rather than failing: there is no
way to observe a finalizer-thread segfault from inside Julia. Hence a process of
its own, whose exit code is the assertion that cannot be made in-process — and
the reset, which replaces the default device, stays out of the suite's.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "a buffer may outlive a device reset" begin
    status, out = infreshprocess(raw"""
        dev = Mantle.Device(BACKEND)
        b = Mantle.backend(dev)
        # Where the backend is among the usable ones: asked again after the reset,
        # that place answers with the default device of its API.
        place = findfirst(==(BACKEND), Mantle.eachbackend())
        @testset "a reset retires, and the new device works" begin
            # A backend array and a pool-backed buffer: each is released by a
            # finalizer, the first through the backend, the second through the
            # pool. Locals of this block, so dropping them below drops the last
            # reference.
            d = KA.allocate(b, Float32, 1000)
            fill!(d, 1.0f0)
            buf = Mantle.Buffer(dev, ones(Float32, 1000))
            KA.synchronize(b)
            @test Array(d) == fill(1.0f0, 1000)
            new = Mantle.reset_device!(dev)
            # A new device, and the default from now on; the old one retired.
            @test new !== dev
            @test Mantle.retired(dev)
            @test !Mantle.retired(new)
            @test Mantle.Device(Mantle.eachbackend()[place]) === new
            # Nothing is allocated on the retired device.
            @test_throws ArgumentError Mantle.Buffer(dev, Float32, 16)
            # The finalizers for the pre-reset buffers, run on purpose. Twice,
            # because the first collection may only queue them.
            d = nothing
            buf = nothing
            GC.gc()
            GC.gc()
            # And the new device computes, which rules out "retired everything".
            b2 = Mantle.backend(new)
            e = KA.allocate(b2, Float32, 64)
            fill!(e, 3.0f0)
            KA.synchronize(b2)
            @test Array(e) == fill(3.0f0, 64)
            f = Mantle.Buffer(new, fill(2f0, 64))
            Mantle.waitidle(new)
            @test Array(Mantle.storage(f)) == fill(2f0, 64)
        end
        """)
    reportfresh(status, out)
    @test status == 0
end
