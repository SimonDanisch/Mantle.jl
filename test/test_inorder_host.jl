using Test,Random
import Mantle
const IOH= Mantle

struct InOrderHostPass
    usages::Vector{Pair{Int,DataType}}
    loop::Union{Nothing,IOH.Loop}
end
mutable struct InOrderHostCompile <: IOH.Compilation
    graph::NamedTuple{(:passes,),Tuple{Vector{InOrderHostPass}}}
    policy::IOH.Policy
    analysis::IOH.Analysis
end
IOH.usages(p::InOrderHostPass)=p.usages
IOH.overlapgroup(::InOrderHostCompile,id::Int)=id
IOH.overlapping(::InOrderHostCompile,a::Int,b::Int)=a==b

@testset "InOrder retains every hazard without quadratic dependencies" begin
    rng=MersenneTwister(1443)
    for _ in 1:100
        ps=[InOrderHostPass([rand(rng,1:8)=>rand(rng,(IOH.CopySrc,IOH.CopyDst))
            for _ in 1:rand(rng,1:4)],nothing) for _ in 1:rand(rng,1:64)]
        baseline=InOrderHostCompile((passes=ps,),IOH.Overlap(),IOH.Analysis())
        ordered=InOrderHostCompile((passes=ps,),IOH.InOrder(),IOH.Analysis())
        IOH.run!(IOH.Dag(),baseline)
        IOH.run!(IOH.Dag(),ordered);IOH.run!(IOH.Schedule(),ordered)
        @test ordered.analysis.order==collect(eachindex(ps))
        # The chain's transitive closure contains EVERY original hazard edge.
        @test all(i<j for j in eachindex(ps) for i in baseline.analysis.deps[j])
        @test ordered.analysis.deps==[i==1 ? Int[] : [i-1] for i in eachindex(ps)]
    end
    n=200_000
    ps=fill(InOrderHostPass([1=>IOH.CopyDst],nothing),n)
    c=InOrderHostCompile((passes=ps,),IOH.InOrder(),IOH.Analysis())
    IOH.run!(IOH.Dag(),c);IOH.run!(IOH.Schedule(),c)
    @test sum(length,c.analysis.deps)==n-1
    @test Base.summarysize(c.analysis.deps)<96n
    @test c.analysis.order==collect(1:n)
    loop=IOH.Loop(3,nothing)
    p=InOrderHostPass(Pair{Int,DataType}[],loop)
    valid=InOrderHostCompile((passes=[p,p,p],),IOH.InOrder(),IOH.Analysis())
    IOH.run!(IOH.Dag(),valid);IOH.run!(IOH.Schedule(),valid)
    @test valid.analysis.order==[1,2,3]
    split=InOrderHostCompile((passes=[p,InOrderHostPass(Pair{Int,DataType}[],nothing),p],),
        IOH.InOrder(),IOH.Analysis())
    IOH.run!(IOH.Dag(),split)
    @test_throws ErrorException IOH.run!(IOH.Schedule(),split)
end
