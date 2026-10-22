using Test, Mantle

mutable struct ExtentProbeDevice <: Mantle.Device
    calls::Int
end
Mantle.maxalloc(::ExtentProbeDevice) = typemax(Int)
Mantle.tryallocate(d::ExtentProbeDevice, ::Type, ::Dims) = (d.calls += 1; nothing)

@testset "byte extents reject invalid shapes before allocation" begin
    huge = 1 << (Sys.WORD_SIZE-2)
    for (dims,bytes) in (((),8),((2,3),48),((0,7),0),((huge,4,0),0))
        @test Mantle.checkedbytes(Float64,dims) == bytes
    end
    @test_throws ArgumentError Mantle.checkedbytes(String,(0,))
    @test_throws ArgumentError Mantle.checkedbytes(Base.RefValue{Int},(1,))
    @test_throws ArgumentError Mantle.checkedbytes(UInt8,(-1,-1))
    @test_throws ArgumentError Mantle.checkedbytes(UInt8,(0,-1))
    @test_throws OverflowError Mantle.checkedbytes(UInt8,(huge,4))
    @test_throws OverflowError Mantle.checkedbytes(Float64,(huge,))
    @test_throws OverflowError Mantle.checkedbytes(Nothing,(huge,4))

    dev = ExtentProbeDevice(0)
    for (T,dims) in ((UInt8,(huge,4)),(Float64,(huge,)))
        @test Mantle.trydevicearray(dev,T,dims) === nothing
        @test_throws OverflowError Mantle.allocate(Mantle.Pool(),dev,:buffer,T,dims)
    end
    @test_throws ArgumentError Mantle.trydevicearray(dev,UInt8,(-1,-1))
    @test_throws ArgumentError Mantle.allocate(Mantle.Pool(),dev,:buffer,UInt8,(-1,-1))
    @test dev.calls == 0
    @test Mantle.trydevicearray(dev,UInt8,(1,)) === nothing
    @test dev.calls == 1 # A representable request reaches the native allocator.

    graph = Mantle.Graph(dev)
    @test_throws OverflowError Mantle.Transient.Buffer(graph,UInt8,(huge,4))
    @test_throws ArgumentError Mantle.Transient.Buffer(graph,UInt8,(-1,-1))
    @test isempty(graph.transients)
end

@testset "host views validate spans without addition overflow" begin
    for (capacity,offset,bytes) in ((8,0,8),(8,8,0),(typemax(Int),typemax(Int)-8,8))
        @test Mantle.checkspan(capacity,offset,bytes) === nothing
    end
    for (capacity,offset,bytes) in ((8,-1,1),(8,0,-1),(-1,0,0),(8,9,0),
                                   (8,7,2),(typemax(Int),typemax(Int),1))
        @test_throws ArgumentError Mantle.checkspan(capacity,offset,bytes)
    end
    data = UInt32[11,22,33,44]
    GC.@preserve data begin
        base = Ptr{UInt8}(pointer(data))
        @test Mantle.hostview(UInt32,base,sizeof(data),4,(2,)) == UInt32[22,33]
        @test isempty(Mantle.hostview(UInt32,base,sizeof(data),sizeof(data),(0,)))
        @test_throws ArgumentError Mantle.hostview(UInt32,base,8,-4,(1,))
        @test_throws ArgumentError Mantle.hostview(UInt32,base,8,typemax(Int),(1,))
        @test_throws ArgumentError Mantle.hostview(UInt32,base,8,0,(-1,-1))
        @test_throws OverflowError Mantle.hostview(UInt32,base,8,0,(1 << (Sys.WORD_SIZE-2),4))
    end
end
