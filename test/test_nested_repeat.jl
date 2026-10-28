using Test, Mantle
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

function nested_add!(out, outer, inner)
    KI.get_global_id().x == 1 || return nothing
    @inbounds out[1] += Int32(10)*outer[1] + inner[1]
    return nothing
end
function nested_start!(flag, left)
    KI.get_global_id().x == 1 || return nothing
    @inbounds begin
        flag[1] = Int32(1)
        left[1] = Int32(2)
    end
    return nothing
end
function nested_countdown!(out, flag, left, index)
    KI.get_global_id().x == 1 || return nothing
    @inbounds begin
        out[1] += index[1]
        left[1] -= Int32(1)
        flag[1] = left[1] > 0 ? Int32(1) : Int32(0)
    end
    return nothing
end
function nested_mark!(out)
    KI.get_global_id().x == 1 || return nothing
    @inbounds out[1] += Int32(100)
    return nothing
end

# A visitor exercises the internal walked-emission order without trying to run
# an unrecorded plan. Public execution always records once before replay.
mutable struct LoopVisitEmitter
    indices::Vector{Int}
end
Mantle.emithead!(::LoopVisitEmitter, plan::Mantle.Plan) = nothing
Mantle.emitloophead!(::LoopVisitEmitter, plan::Mantle.Plan, range) = nothing
Mantle.emitpasses!(e::LoopVisitEmitter, plan::Mantle.Plan, range) = (append!(e.indices,range);nothing)

@testset "Nested native repeat scheduling and recorded replay" begin
    dev = Mantle.Device(TESTBACKEND)
    for policy in (Mantle.Compact(),Mantle.Overlap(),Mantle.InOrder()), budget in (Inf,.01)
        out = Mantle.Buffer(dev,Int32[0])
        graph = Mantle.Graph(dev)
        Mantle.dispatch!(graph,Mantle.fill_kernel!,(out,Int32(0)),1)
        Mantle.repeat!(graph,3) do outer
            Mantle.repeat!(graph,4) do inner
                Mantle.dispatch!(graph,nested_add!,(out,outer,inner),1)
            end
        end
        plan = Mantle.Plan(graph;policy,budget)
        try
            visit = LoopVisitEmitter(Int[])
            Mantle.emitplan!(visit,plan)
            names = [plan.passes[i].pass.name for i in visit.indices]
            @test count(==("nested_add!"),names) == 12
            @test count(==("repeat!/start"),names) == 4
            @test count(==("repeat!/advance"),names) == 15
            Mantle.record!(plan)
            @test length(plan.loops) == 2
            for run in 1:3
                Mantle.run!(plan); Mantle.waitfor!(plan)
                @test Array(out) == Int32[270]
            end
        finally
            Mantle.free!(plan)
        end
    end
end

if Mantle.supportspredicate(Mantle.Device(TESTBACKEND))
@testset "Nested device gates clear stale child predicates" begin
    dev = Mantle.Device(TESTBACKEND)
    for policy in (Mantle.Compact(),Mantle.Overlap(),Mantle.InOrder()), budget in (Inf,.01)
        graph = Mantle.Graph(dev)
        outer_count = Mantle.GPURef(dev,Int32(0))
        inner_active = Mantle.Buffer(dev,Int32[1])
        left = Mantle.Buffer(dev,Int32[0])
        output = Mantle.Buffer(dev,Int32[0])
        Mantle.dispatch!(graph,Mantle.fill_kernel!,(output,Int32(0)),1)
        Mantle.repeat!(graph,3,outer_count) do outer
            Mantle.dispatch!(graph,nested_start!,(inner_active,left),1)
            Mantle.repeat!(graph,5;while_nonzero=inner_active) do inner
                Mantle.dispatch!(graph,nested_countdown!,(output,inner_active,left,inner),1)
            end
            # This sibling tests the ordering out of the inner loop.
            Mantle.dispatch!(graph,nested_mark!,(output,),1)
        end
        plan = Mantle.record!(Mantle.Plan(graph;policy,budget))
        try
            for n in (2,0,1,3,0)
                outer_count[] = Int32(n)
                # A skipped outer loop must clear the child gate even if both
                # its source flag and its last predicate were left nonzero.
                Mantle.update!(inner_active,Int32[1])
                Mantle.run!(plan); Mantle.waitfor!(plan)
                @test Array(output) == Int32[103n]
            end
        finally
            Mantle.free!(plan)
        end
    end
end
end

if Mantle.supportspredicate(Mantle.Device(TESTBACKEND))
@testset "Three nested gates reset indices and preserve transient lifetimes" begin
    dev = Mantle.Device(TESTBACKEND)
    for policy in (Mantle.Compact(),Mantle.Overlap(),Mantle.InOrder())
        graph = Mantle.Graph(dev)
        count = Mantle.GPURef(dev,Int32(0))
        middle_count = Mantle.GPURef(dev,Int32(2))
        active = Mantle.Buffer(dev,Int32[1])
        left = Mantle.Buffer(dev,Int32[0])
        output = Mantle.Buffer(dev,Int32[0])
        work = Mantle.Transient.Buffer(graph,Int32,1)
        Mantle.dispatch!(graph,Mantle.fill_kernel!,(work,Int32(0)),1)
        Mantle.repeat!(graph,3,count) do outer
            Mantle.repeat!(graph,4,middle_count) do middle
                Mantle.dispatch!(graph,nested_start!,(active,left),1)
                Mantle.repeat!(graph,4;while_nonzero=active) do inner
                    Mantle.dispatch!(graph,nested_countdown!,(work,active,left,inner),1)
                end
            end
        end
        Mantle.dispatch!(graph,Mantle.d2dcopy_kernel!,(output,work,Int64(0),Int64(0),Int64(1)),1)
        plan = Mantle.record!(Mantle.Plan(graph;policy,budget=Inf))
        try
            @test length(plan.loops) == 3
            for n in (2,0,1,3,0)
                count[] = Int32(n)
                Mantle.update!(active,Int32[1])
                Mantle.run!(plan); Mantle.waitfor!(plan)
                @test Array(output) == Int32[6n]
            end
        finally
            Mantle.free!(plan)
        end
    end
end
end
