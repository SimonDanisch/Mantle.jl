"""
A backend knows which device it runs on, and pins it when it is built.

The Vulkan backend's `LavaBackend()` stored `nothing` in both queue fields and
resolved the process default at every property access, which is what let a
module-level `const BACKEND = LavaBackend()` survive `reset_device!`. That made
"which device" a global lookup on every launch, and a backend handed to code that
also held a second device dispatched on whichever was current. Every spelling
pins now: a backend names its device, an array names the backend it was
allocated on, and the two agree.

What a user sees when this breaks is a kernel, an allocation or a per-device
cache landing on a device other than the one the caller handed over; so this
asserts the round trips between the three handles a caller can hold — a
device, its backend and an array — rather than the fields behind them.
"""

using Test, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "a backend knows its device" begin
    dev = Mantle.Device(TESTBACKEND)

    # ── Every handle resolves to the device it dispatches on: the device's
    #    backend, and the backend under test, name the same device object.
    @test Mantle.Device(TESTBACKEND) === dev
    @test Mantle.Device(Mantle.backend(dev)) === dev
    @test Mantle.backend(dev) == TESTBACKEND
    @test Mantle.todevice(TESTBACKEND) === dev

    # ── The round trip that makes it useful: an array knows its device, and the
    #    backend derived from it agrees. This is the path a per-device cache key
    #    travels.
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 8))
    @test KA.get_backend(a) == TESTBACKEND
    @test Mantle.Device(KA.get_backend(a)) === dev

    # ── And it still works as a backend.
    fill!(a, 2.0f0)
    KA.synchronize(KA.get_backend(a))
    @test Array(a) == fill(2.0f0, 8)
end
