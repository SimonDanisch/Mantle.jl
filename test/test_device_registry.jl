# Core's device table (`runtime/lifecycle.jl`), which every backend's default device
# goes through: one default however many tasks ask for it at once, every device made
# is retired at exit, and a reset hands the default to the replacement.
#
# Each backend used to keep its own: Vulkan's behind a lock, Metal's and ROCm's
# without one, so two tasks asking for the first Metal device at once could build
# two, with two pools over the same memory. Asked of a test API here, whose
# `makedevice` yields part way, so concurrent tasks interleave even on one thread.

using Test, Mantle

struct RegistryAPI <: Mantle.Backend end

mutable struct RegistryDevice <: Mantle.Device
    debug::Mantle.DebugConfig
    retired::Bool
    adopted::Int
end

const MADE = Ref(0)

function Mantle.makedevice(::RegistryAPI, select, debug::Mantle.DebugConfig)
    MADE[] += 1
    sleep(0.01)
    return RegistryDevice(debug, false, 0)
end
Mantle.remakedevice(::RegistryDevice, debug::Mantle.DebugConfig) =
    Mantle.makedevice(RegistryAPI(), nothing, debug)
Mantle.syncbackend(::RegistryDevice) = RegistryAPI()
Mantle.retire!(d::RegistryDevice) = (d.retired = true; nothing)
Mantle.retired(d::RegistryDevice) = d.retired
Mantle.adopt!(d::RegistryDevice) = (d.adopted += 1; d)

Mantle.initdevices!((RegistryAPI,))

@testset "core's device table" begin
    @testset "one default however many tasks ask at once" begin
        tasks = [Threads.@spawn Mantle.Device(RegistryAPI()) for _ in 1:16]
        devs = fetch.(tasks)
        @test MADE[] == 1
        @test all(d -> d === devs[1], devs)
        # Adopted once, when it became the default.
        @test devs[1].adopted == 1
    end

    @testset "a selected or configured device is new and not the default" begin
        d = Mantle.Device(RegistryAPI(); debug = Mantle.DebugConfig(validation = true))
        @test d !== Mantle.Device(RegistryAPI())
        @test d.debug.validation
        @test any(x -> x === d, Mantle.livedevices(RegistryAPI()))
    end

    @testset "a reset retires the device and hands its replacement the default" begin
        old = Mantle.Device(RegistryAPI())
        new = Mantle.reset_device!(old)
        @test Mantle.retired(old) && !Mantle.retired(new)
        @test Mantle.Device(RegistryAPI()) === new
        # The configuration carries over unless the reset says otherwise.
        @test new.debug == old.debug
        @test Mantle.reset_device!(new; debug = Mantle.DebugConfig(validation = true)).debug.validation
    end

    @testset "the exit hook retires every device made" begin
        live = Mantle.livedevices(RegistryAPI())
        @test length(live) >= 3
        Mantle.retireall!(RegistryAPI())
        @test all(Mantle.retired, live)
        @test Mantle.existingdefault(RegistryAPI()) === nothing
    end
end
