# Evaluate the actual shared fixture/caps helpers without loading Mantle.
module SuiteHelperContractTests
using Test
module Mantle
abstract type Device end
function name end
function channelinfo end
function queuekinds end
end
struct MockDevice <: Mantle.Device
    facts::NamedTuple
    channels::Vector{NamedTuple}
end
caps(d::MockDevice) = d.facts
Mantle.channelinfo(d::MockDevice) = d.channels
Mantle.queuekinds(d::MockDevice) = (:main, :compute, :transfer)
Mantle.name(d::MockDevice) = d.facts.software ? "llvmpipe" : "GPU"

const wanted = (:testdevice, :has, :defaultcandidate, :vulkanpeers, :withcaps, :replacecaps)
function install(ex)
    ex isa Expr || return
    if ex.head === :struct && ex.args[2] === :SuiteConfig
        Core.eval(@__MODULE__, ex)
        return
    elseif ex.head === :const && ex.args[1] isa Expr && ex.args[1].args[1] === :SUITE
        Core.eval(@__MODULE__, ex)
        return
    elseif ex.head in (:function, :(=))
        sig = ex.args[1]
        while sig isa Expr && sig.head === :where
            sig = sig.args[1]
        end
        if sig isa Expr && sig.head === :call && sig.args[1] in wanted
            Core.eval(@__MODULE__, ex)
            return
        end
    end
    foreach(install, ex.args)
end
install(Meta.parseall(read(joinpath(@__DIR__, "..", "suite", "helpers.jl"), String)))

struct StructCaps
    bandwidth::Float64
    display::Bool
    displayattached::Bool
end
@testset "Shared caps and fixture helpers" begin
    c = (; bandwidth = 100.0, display = false, displayattached = true)
    changed = withcaps(c; bandwidth = 1.0)
    @test changed.bandwidth == 1.0 && c.bandwidth == 100.0
    @test changed.display === false && changed.displayattached === true
    @test_throws ArgumentError withcaps(c; bandwith = 1.0)
    sc = withcaps(StructCaps(100.0, false, true); displayattached = false)
    @test sc.bandwidth == 100.0 && !sc.display && !sc.displayattached
    @test_throws ArgumentError withcaps(sc; missing = 1)

    state1, state2 = Ref(1), Ref(2)
    facts = (; api = :vulkan, software = true, graphics = true, raytracing = false,
             sparseresidency = false, display = false)
    # Three queue labels alias the same native queue: no cross-queue fixture.
    cpu = MockDevice(facts, [(; state = state1, compute = true, transfer = true),
                             (; state = state1, compute = true, transfer = true)])
    SUITE.fixtures[] = (; devices = (cpu,), driverpairs = ())
    SUITE.device[] = cpu
    @test testdevice() === cpu
    @test has(cpu, :nodefaultgpu) && !has(cpu, :defaultgpu)
    @test !has(cpu, :twodevices)
    @test !has(cpu, :multiqueue) && !has(cpu, :multicompute) && !has(cpu, :transferqueue)
    @test has(cpu, :lavapipe) && !has(cpu, :window)

    gpu = MockDevice(merge(facts, (; software = false)),
                     [(; state = state1, compute = true, transfer = true),
                      (; state = state2, compute = false, transfer = true)])
    @test has(gpu, :multiqueue) && has(gpu, :transferqueue)
    @test !has(gpu, :multicompute)
    secondcompute = MockDevice(gpu.facts,
                     [(; state = state1, compute = true, transfer = true),
                      (; state = state2, compute = true, transfer = true)])
    @test has(secondcompute, :multicompute)
    cu = MockDevice(merge(facts, (; api = :cuda, software = false)), gpu.channels)
    SUITE.fixtures[] = (; devices = (cpu, gpu, cu), driverpairs = ((cu,gpu),))
    @test has(cpu, :defaultgpu) && !has(cpu, :nodefaultgpu)
    @test has(cpu, :twodevices)
    @test vulkanpeers(cu) == [gpu] && has(cu, :vulkanpeer)
    @test !has(cpu, :vulkanpeer)
    @test has(cu, :vendorview) && !has(gpu, :vendorview)
    @test !has(cpu, :metal)
end
end
