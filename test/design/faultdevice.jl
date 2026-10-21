# Execute the actual suite instrument against a tiny native ABI, without
# importing the current Mantle implementation or constructing a GPU device.
module FaultDeviceContractTests
using Test
module Mantle
abstract type Device end
abstract type SubmitChannel end
struct Runtime
    owner::Device
    allocated::Base.RefValue{Int}
end
Runtime(owner) = Runtime(owner, Ref(0))
struct Window{D}; device::D; inner; end
struct Swapchain{D}; device::D; inner; end
device(x::Union{Window,Swapchain}) = x.device
inner(x::Union{Window,Swapchain}) = x.inner
end
# Declare precisely the generic verbs referenced by the instrument. No stub
# supplies a catch-all forwarding implementation: missing interception fails.
const source = read(joinpath(@__DIR__, "..", "suite", "faultdevice.jl"), String)
const verbs = unique([Symbol(m.captures[1]) for m in eachmatch(r"Mantle\.([A-Za-z_][A-Za-z_0-9!]*)(?:\(|\b)", source)])
append!(verbs, [Symbol(m.captures[1]) for m in eachmatch(r"@forward(?:channel|builder|window)?\s+([A-Za-z_][A-Za-z_0-9!]*)", source)])
for v in unique(verbs)
    isdefined(Mantle, v) || Core.eval(Mantle, :(function $v end))
end
include("../suite/faultdevice.jl")
Mantle.reserved(d::FaultDevice) = Mantle.runtime(d).allocated[]

mutable struct State
    next::Int
    lock::ReentrantLock
end
struct NativeChannel <: Mantle.SubmitChannel
    state::State
    calls::Vector{Symbol}
end
struct NativeDevice <: Mantle.Device
    ch::NativeChannel
end
struct NativeBuilder
    calls::Vector{Symbol}
end
Mantle.channelstate(ch::NativeChannel) = ch.state
Mantle.openchannels(d::NativeDevice, kinds) = fill(d.ch, length(kinds))
Mantle.foreignchannel(d::NativeDevice, stream) = d.ch
Mantle.Builder(d::NativeDevice, ch::NativeChannel) = NativeBuilder(ch.calls)
for v in (:preparesubmit, :submitnative!, :passed, :waitfor, :waitnativeidle!, :recycle!,
          :bindpages!, :unbindpages!, :presentnative!, :streamwaits!, :signal!)
    @eval function Mantle.$v(ch::NativeChannel, args...; kw...)
        push!(ch.calls, $(QuoteNode(v)))
        return (; verb = $(QuoteNode(v)), args, kw = (; kw...))
    end
end
for v in (:emittimestamp!, :emitdispatch!, :emittrace!, :emitcopy!, :emitstore!, :emitbuild!,
          :beginrender!, :emitpieces!, :endrender!, :bindindices!, :emitdraw!,
          :emitlibrary!, :emitdecode!, :finish)
    @eval function Mantle.$v(b::NativeBuilder, args...; kw...)
        push!(b.calls, $(QuoteNode(v)))
        return (args, (; kw...))
    end
end
Mantle.recordpiece(d::NativeDevice, node, region, emit) = emit(NativeBuilder(d.ch.calls))
Mantle.presentpassed(record::Base.RefValue{Bool}) = record[]

@testset "FaultDevice native boundary" begin
    native = NativeDevice(NativeChannel(State(0, ReentrantLock()), Symbol[]))
    a, b = FaultDevice(native), FaultDevice(native)
    @test Mantle.runtime(a) !== Mantle.runtime(b)
    @test Mantle.runtime(a).owner === a
    ch, alias = Mantle.openchannels(a, (:main, :transfer))
    other = only(Mantle.openchannels(b, (:main,)))
    @test Mantle.device(ch) === a
    @test Mantle.channelstate(ch) === Mantle.channelstate(alias) === Mantle.channelstate(other)
    @test Mantle.channelstate(Mantle.foreignchannel(a, :stream)) === Mantle.channelstate(ch)
    for v in (:preparesubmit, :submitnative!, :passed, :waitfor, :waitnativeidle!, :recycle!,
              :bindpages!, :unbindpages!, :presentnative!, :streamwaits!, :signal!)
        result = getproperty(Mantle, v)(ch, 7; tag = :sentinel)
        @test result == (; verb = v, args = (7,), kw = (; tag = :sentinel))
        @test calls(a, v) == 1
    end
    before = length(native.ch.calls)
    inject!(a, :preparesubmit, Throw(2, ErrorException("preclaim")))
    @test_throws ErrorException Mantle.preparesubmit(ch, :packet)
    @test calls(a, :preparesubmit) == 2
    @test length(native.ch.calls) == before
    @test Mantle.preparesubmit(ch, :retry).args == (:retry,)
    @test_throws ArgumentError inject!(a, :submitnative!, Throw(2, ErrorException("lost")))
    isolated = FaultDevice(native; isolated = true) # mock endpoints only
    isolatedch = only(Mantle.openchannels(isolated, (:main,)))
    inject!(isolated, :submitnative!, Throw(1, ErrorException("lost")))
    @test_throws ErrorException Mantle.submitnative!(isolatedch, :packet)
    @test calls(isolated, :submitnative!) == 1
    builder = Mantle.Builder(a, ch)
    @test Mantle.device(builder) === a
    for v in (:emittimestamp!, :emitdispatch!, :emittrace!, :emitcopy!, :emitstore!, :emitbuild!,
              :beginrender!, :emitpieces!, :endrender!, :bindindices!, :emitdraw!,
              :emitlibrary!, :emitdecode!, :finish)
        @test getproperty(Mantle, v)(builder, 9; tag = :sentinel) == ((9,), (; tag = :sentinel))
        @test calls(a, v) == 1
    end
    @test Mantle.recordpiece(a, :node, :region, b -> Mantle.emitdraw!(b, 11)) == ((11,), (;))
    @test calls(a, :recordpiece) == 1
    @test calls(a, :emitdraw!) == 2
    complete = Ref(false)
    record = Mantle.nativeowner(ch, complete)
    @test !Mantle.presentpassed(record)
    complete[] = true
    @test Mantle.presentpassed(record)
    @test calls(a, :presentpassed) == 2
    inject!(a, :passed, AnswerOnce(2, false))
    @test Mantle.passed(ch, 1) === false
    @test Mantle.passed(ch, 1).verb === :passed
    gate = Channel{Nothing}(1)
    inject!(a, :submitnative!, Block(2, gate))
    task = @async Mantle.submitnative!(ch, :blocked)
    yield()
    @test !istaskdone(task)
    put!(gate, nothing)
    @test fetch(task).args == (:blocked,)
    @test calls(a, :submitnative!) == 2
    Mantle.runtime(a).allocated[] = 400
    inject!(a, :budget, BudgetLimit(1, a, 1000))
    @test Mantle.budget(a, :heap) == 600
    Mantle.runtime(a).allocated[] = 700
    @test Mantle.budget(a, :heap) == 300
    Mantle.runtime(a).allocated[] = 1200
    @test Mantle.budget(a, :heap) == 0
    @test calls(a, :budget) == 3
end
end
