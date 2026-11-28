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

using Test, KernelAbstractions, Raycore, Adapt
import GeometryBasics
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "a backend equals every spelling of its device" begin
    dev = Mantle.Device(TESTBACKEND)
    @test TESTBACKEND == dev
    @test hash(TESTBACKEND) == hash(dev)

    # An array allocated by the backend belongs to it, whichever spelling asks,
    # also where the array reports its array package's backend (Metal.jl's for an
    # `MtlArray`).
    a = KA.allocate(TESTBACKEND, Float32, 8)
    @test KA.get_backend(a) == TESTBACKEND
    @test TESTBACKEND == KA.get_backend(a)
    @test Mantle.Device(KA.get_backend(a)) === dev
    b = Mantle.devicearray(dev, Float32, 8)
    @test KA.get_backend(b) == KA.get_backend(a)
end

# Raycore refuses to adapt a TLAS to another device's backend. A renderer adapts
# its scene with the backend its film's arrays report, which on Metal is Metal.jl's
# backend for the device the TLAS was built on: RayMakie's meshscatter scenes
# failed with "Cross-backend Adapt.adapt(::MetalBackend, ::TLAS)" until that
# backend equalled its device.
@testset "a TLAS adapts to the backend its device's arrays report" begin
    tlas = Raycore.TLAS(TESTBACKEND)
    push!(tlas, GeometryBasics.normal_mesh(GeometryBasics.Rect3f(GeometryBasics.Point3f(0),
                                                                 GeometryBasics.Vec3f(1))))
    Raycore.sync!(tlas)
    film = KA.allocate(TESTBACKEND, Float32, 4)
    @test Adapt.adapt(KA.get_backend(film), tlas) === tlas.static_tlas
end
