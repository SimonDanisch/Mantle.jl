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

using Test, KernelAbstractions, Adapt
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function backend_context_twice!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = 2f0 * x[i]
end

@testset "a backend knows its device" begin
    dev = Mantle.Device(TESTBACKEND)

    # ── A registered backend IS its device. On Metal `defaultbackend()` handed
    #    out Metal.jl's `MetalBackend` instead, which no device verb answers:
    #    `Mantle.devicename(Mantle.defaultbackend())` threw a `MethodError`.
    @test TESTBACKEND isa Mantle.Device
    @test all(b -> b isa Mantle.Device, Mantle.eachbackend())
    @test Mantle.devicename(TESTBACKEND) isa String

    # ── Every handle resolves to the device it dispatches on: the device's
    #    backend, and the backend under test, name the same device object.
    @test TESTBACKEND === dev
    @test Mantle.todevice(TESTBACKEND) === dev

    # ── The round trip that makes it useful: an array knows its device, and the
    #    backend derived from it agrees. This is the path a per-device cache key
    #    travels. An array may report its array package's backend (Metal.jl's for
    #    an `MtlArray`), which equals the device it lives on.
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 8))
    @test KA.get_backend(a) == TESTBACKEND
    @test TESTBACKEND == KA.get_backend(a)
    @test Mantle.Device(KA.get_backend(a)) === dev
    @test Mantle.caps(KA.get_backend(a)) == Mantle.caps(dev)
    @test Mantle.supports_hwtlas(KA.get_backend(a)) == Mantle.supports_hwtlas(dev)

    # ── And it still works as a backend: the array's, and the device itself.
    fill!(a, 2.0f0)
    KA.synchronize(KA.get_backend(a))
    @test Array(a) == fill(2.0f0, 8)

    x = Adapt.adapt(dev, Float32.(1:64))
    y = KA.zeros(dev, Float32, 64)
    @test Mantle.Device(KA.get_backend(y)) === dev
    backend_context_twice!(dev)(y, x; ndrange = 64)
    KA.synchronize(dev)
    @test Array(y) == 2f0 .* (1:64)
end
