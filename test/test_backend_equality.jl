"""
A backend equals every spelling of its DEVICE.

`==` must not compare queues: with the Vulkan backend's `LavaBackend()` resolving
its queues at each access and `LavaBackend(ctx)` pinning them, the default `==`
compared `nothing` against a queue, so an array allocated by the default backend
did not belong to it as far as `==` could tell (RayMakie's meshscatter tests
failed on exactly that).

Now `==` is "the same device". That is the question both callers ask: Raycore
refuses a cross-backend `adapt` of a TLAS, and RayMakie asks whether an array is
the screen's. Two devices are two backends, which `test_device_identity.jl`
asserts against a second device.

A backend on a SECOND queue of the same device answering the same way is not
asserted here: there is no portable way to build a backend on a queue from
`allocate_batch_queue!` (see the integrator notes for this file).
"""

using Test, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "a backend equals every spelling of its device" begin
    dev = Mantle.Device(TESTBACKEND)
    pinned = Mantle.backend(dev)
    @test TESTBACKEND == pinned
    @test pinned == TESTBACKEND
    @test hash(TESTBACKEND) == hash(pinned)

    # An array allocated by the backend belongs to it, whichever spelling asks.
    a = KA.allocate(TESTBACKEND, Float32, 8)
    @test KA.get_backend(a) == TESTBACKEND
    @test KA.get_backend(a) == pinned
    @test hash(KA.get_backend(a)) == hash(pinned)
    b = Mantle.devicearray(dev, Float32, 8)
    @test KA.get_backend(b) == KA.get_backend(a)
end
