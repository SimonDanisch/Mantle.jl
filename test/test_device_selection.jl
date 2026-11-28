"""
Choosing a device: the listing, the selector vocabulary, and the default.

`Device(select)` and `defaultdevice!(select)` replace pinning the loader to one ICD
with `VK_DRIVER_FILES`, which also hid every other device from these tests, and
the `MANTLE_DEVICE` variable after it: a device is chosen in code. `selectdevice` is where a name substring, an index, a predicate
and `nothing` all become one physical-device index, ranked discrete-first, so the
answer is the same on every backend.
"""

using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "device selection" begin
    infos = Mantle.devices()
    @test !isempty(infos)
    @test all(i -> i.index isa Int && !isempty(i.name), infos)

    # A synthetic listing so the vocabulary is checked without depending on the
    # machine's exact devices.
    dev = [Mantle.DeviceInfo(1, "NVIDIA RTX 4000 Ada", :discrete, "NVIDIA"),
           Mantle.DeviceInfo(2, "AMD Radeon RX 7900", :discrete, "radv"),
           Mantle.DeviceInfo(3, "llvmpipe", :cpu, "llvmpipe")]
    @test Mantle.selectdevice(nothing, dev) == 1          # discrete, first in order
    @test Mantle.selectdevice("radeon", dev) == 2         # name, case-insensitive
    @test Mantle.selectdevice("NVIDIA", dev) == 1         # driver, case-insensitive
    @test Mantle.selectdevice("lavapipe", [
        Mantle.DeviceInfo(1, "llvmpipe", :cpu, "lavapipe llvmpipe Mesa")
    ]) == 1
    @test Mantle.selectdevice(3, dev) == 3                # index
    @test Mantle.selectdevice(i -> i.kind == :cpu, dev) == 3
    @test_throws ArgumentError Mantle.selectdevice("nonexistent", dev)

    # The ranking: a cpu device is chosen last, even enumerated first.
    ranked = [Mantle.DeviceInfo(1, "llvmpipe", :cpu, "x"),
              Mantle.DeviceInfo(2, "APU", :integrated, "y"),
              Mantle.DeviceInfo(3, "dGPU", :discrete, "z")]
    @test Mantle.selectdevice(nothing, ranked) == 3
    @test Mantle.selectdevice("", ranked) == 3            # empty string admits all, then ranks
end

@testset "a device chosen by selector, and made the default" begin
    infos = Mantle.devices()
    i = Mantle.selectdevice(nothing, infos)
    old = Mantle.Device()
    # A device of your own: a context of its own, not the default.
    mine = Mantle.Device(i)
    @test mine isa Mantle.Device
    @test mine !== old
    @test Mantle.Device() === old
    # …and in one step installed and returned. The old default stays a device
    # of its own, so putting it back is another `defaultdevice!`.
    new = Mantle.defaultdevice!(i)
    @test new !== old
    @test Mantle.Device() === new
    @test Mantle.defaultdevice!(old) === old
    @test Mantle.Device() === old
end
