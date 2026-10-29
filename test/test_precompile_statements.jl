using Test, Mantle

module MantlePrecompileDeviceProbe
using Mantle
struct API <: Mantle.Backend end
struct Device <: Mantle.Device end
const constructions = Ref(0)
function Mantle.makedevice(::API, select, debug::Mantle.DebugConfig)
    constructions[] += 1
    return Device()
end
end

@testset "host precompilation constructs no devices" begin
    @test MantlePrecompileDeviceProbe.Device in Mantle._precompile_device_types()
    @test MantlePrecompileDeviceProbe.constructions[] == 0
    ok, total = Mantle._precompile_statements()
    @test total > 0
    @test ok == total
    @test MantlePrecompileDeviceProbe.constructions[] == 0
end
