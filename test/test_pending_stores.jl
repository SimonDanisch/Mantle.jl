using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "pending range stores share ordering and failure contracts" begin
    dev = Mantle.Device(TESTBACKEND)
    b = Mantle.Buffer(dev, Int32[]; capacity=6)
    t = Mantle.TransientBuffer{Int32,2}((2,3),typemax(Int),0,nothing,0;
                                      hostwritten=true)
    try
        for r in (b,t)
            data = Int32[3,4]
            @test setindex!(r,data,2:3) === r
            @test r.pending[1][2] === data
            r[3:4] = Int16[7,8]
            @test r.pending[2][2] == Int32[7,8]
            @test_throws DimensionMismatch setindex!(r,data,1:1)
            @test_throws ArgumentError setindex!(r,data,0:1)
            @test_throws ArgumentError setindex!(r,data,6:7)
            @test length(r.pending) == 2
            seen = Tuple{Int,Vector{Int32}}[]
            failsecond = (_,v,i) -> (i == 3 && error("store callback"); push!(seen,(i,v)))
            @test_throws ErrorException Mantle.landstores!(failsecond,r)
            @test Mantle.isdirty(r) && length(r.pending) == 2
            @test seen == [(2,data)]
            @test r isa Mantle.Buffer ? r.len == 3 : size(r) == (2,3)
            Mantle.landstores!((_,v,i)->push!(seen,(i,v)),r)
            @test first.(seen) == [2,2,3]
            @test isempty(r.pending) && !Mantle.isdirty(r)
            @test r isa Mantle.Buffer ? r.len == 4 : size(r) == (2,3)
            Mantle.landstores!((args...)->error("clean store called"),r)

            # A setter arriving during consumption must remain pending for the next drain.
            r[1:1] = Int32[11]
            entered, release, attempted = Channel{Nothing}(1),Channel{Nothing}(1),Channel{Nothing}(1)
            consumed = Int32[]
            drain = Threads.@spawn Mantle.landstores!(r) do _,v,i
                if i == 1
                    put!(entered,nothing)
                    take!(release)
                end
                append!(consumed,v)
            end
            take!(entered)
            setter = Threads.@spawn begin
                put!(attempted,nothing)
                r[2:2] = Int32[22]
            end
            take!(attempted)
            put!(release,nothing)
            fetch(drain); fetch(setter)
            @test consumed == [11]
            @test Mantle.isdirty(r) && length(r.pending) == 1
            Mantle.landstores!((_,v,i)->append!(consumed,v),r)
            @test consumed == [11,22]
            @test !Mantle.isdirty(r)
        end
        undeclared = Mantle.TransientBuffer{Int32,1}((6,),typemax(Int),0,nothing,0)
        @test_throws ArgumentError setindex!(undeclared,Int32[1],1:1)
        @test isempty(undeclared.pending) && !Mantle.isdirty(undeclared)
    finally
        Mantle.free!(b)
    end
end

@testset "mixed inline and large range stores retain declaration order" begin
    dev = Mantle.Device(TESTBACKEND)
    for transient in (false,true)
        n = 20_000
        g = Mantle.Graph(dev)
        src = transient ? Mantle.Transient.Buffer(g,Float32,n;hostwritten=true) :
                          Mantle.Buffer(dev,zeros(Float32,n))
        out = Mantle.Buffer(dev,zeros(Float32,n))
        Mantle.dispatch!(g,Mantle.d2dcopy_kernel!,
                         (out,src,Int64(0),Int64(0),Int64(n)),n;name="copy stores")
        pl = Mantle.Plan(g;budget=Inf)
        try
            Mantle.record!(pl)
            for value in (3f0,7f0)
                src[1:64] = fill(-1f0,64)
                src[:] = fill(value,n) # exceeds vkCmdUpdateBuffer's 64 KiB limit
                src[33:96] = fill(11f0,64)
                src[1:0] = Float32[]
                Mantle.run!(pl)
                Mantle.waitfor!(pl)
                expected = fill(value,n); expected[33:96] .= 11f0
                @test Array(out) == expected
                @test !Mantle.isdirty(src)
            end
        finally
            Mantle.free!(pl)
            transient || Mantle.free!(src)
            Mantle.free!(out)
        end
    end
end
