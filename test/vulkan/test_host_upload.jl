using Test, Mantle, GPUArrays, StaticArrays

function check_host_upload(ctx)
    @testset "typed uploads preserve shape order, bits and offsets" begin
        bq = ctx.default_bq
        dense = reshape(Float32.(1:6),2,3)
        sources = (fill(9f0),Float32[],dense, view(Float32.(1:12),2:2:12), PermutedDimsArray(dense,(2,1)),
                   reshape(UInt8.(1:6),3,2), Float16.(dense), ComplexF32.(dense),
                   reinterpret(Float32,UInt32[0,0x80000000,0x7fc01234,0x7f800000]),
                   BitArray([true,false,true]),SVector(7f0,8f0))
        for unified in (false,true), source in sources
            dst = Mantle.LavaArray{eltype(source)}(undef,(length(source),);bq,unified)
            @test Mantle.upload!(dst,source) === nothing
            @test collect(reinterpret(UInt8,Array(dst))) ==
                  collect(reinterpret(UInt8,vec(collect(source))))
            Mantle.unsafe_free!(dst)
        end
        parent = Mantle.LavaArray(fill(-1f0,14);bq)
        window = GPUArrays.derive(Float32,parent,(6,),3)
        @test Mantle.upload!(window,dense) === nothing
        expected = fill(-1f0,14); expected[4:9] .= vec(dense)
        @test Array(parent) == expected
        @test_throws AssertionError Mantle.upload!(window,zeros(Float32,7))
        @test Array(parent) == expected
        @test_throws ArgumentError Mantle.upload_typed!(parent.buf[],["not device values"])
        Mantle.upload_typed!(parent.buf[],reshape(Float32[9,8],1,2);
                             offset=parent.offset+10sizeof(Float32))
        expected[11:12] = Float32[9,8]
        @test Array(parent) == expected
        Mantle.upload_typed!(parent.buf[],Float32[0,9,8,0],UInt32(2),UInt64(2);offset=parent.offset)
        expected[1:2] = Float32[9,8]
        @test Array(parent) == expected
        bytes = collect(reinterpret(UInt8,Float32[7,6]))
        Mantle.upload!(parent.buf[],bytes;offset=parent.offset)
        expected[1:2] = Float32[7,6]
        @test Array(parent) == expected
        source = ones(Float32,1<<18)
        for unified in (false,true)
            dst = Mantle.LavaArray{Float32}(undef,size(source);bq,unified)
            Mantle.upload!(dst,source); Mantle.upload!(dst,source)
            allocated = @allocated Mantle.upload!(dst,source)
            @test allocated < sizeof(source)÷4
            @test Array(dst) == source
            unified && @test dst.buf[].mapped_ptr != C_NULL
            Mantle.unsafe_free!(dst)
        end
        empty = Mantle.LavaArray{Float32}(undef,(0,);bq)
        @test Mantle.upload!(empty,Float32[]) === nothing
        @test isempty(Array(empty))
        foreach(Mantle.unsafe_free!,(window,parent,empty))
        bytes=zeros(UInt8,32*1024*1024+8)
        bytes[5:8]=UInt8[0x19,0x40,0x80,0x7f];bytes[end-4]=0x70
        words=reinterpret(UInt32,view(bytes,5:length(bytes)-4))
        gpu=Mantle.LavaArray{UInt32}(undef,(length(words)+2,);bq)
        try
            fill!(gpu,UInt32(0x12345678))
            target=GPUArrays.derive(UInt32,gpu,(length(words),),1)
            Mantle.upload!(target,words);copyto!(target,1,words,1,length(words));copyto!(target,words)
            @test (@allocated Mantle.upload!(target,words))<1024*1024
            @test (@allocated copyto!(target,1,words,1,length(words)))<1024*1024
            @test (@allocated copyto!(target,words))<1024*1024
            @test_throws BoundsError copyto!(target,1,words,0,1)
            @test_throws BoundsError copyto!(target,length(target)+1,words,1,1)
            GC.gc()
            result=Array(gpu)
            @test result[2:end-1]==words
            @test first(result)==last(result)==UInt32(0x12345678)
        finally
            Mantle.unsafe_free!(gpu)
        end
        Mantle.waitidle(Mantle.lavadevice(ctx)); Mantle.drain!(bq)
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end

function check_store_submission(ctx)
    @testset "host stores stay in the caller's submission" begin
        dev = Mantle.lavadevice(ctx)
        for (off,n) in ((0,0),(4,16),(4,65536),(4,65540),(1,7))
            dst = Mantle.Buffer(dev,zeros(UInt8,n+8))
            src = fill(UInt8(0x73),n)
            try
                before = Mantle.submissions(dev)
                o = Mantle.oneshot(dev.bq) do e
                    GC.@preserve src Mantle.storebytes!(e,dst.store,off,
                                                        Ptr{Cvoid}(pointer(src)),n)
                end
                @test Mantle.submissions(dev) == before
                Mantle.release!(dev.bq,o)
                @test all(iszero,Array(dst)) # discarding the unsubmitted store writes nothing
                Mantle.oneshot!(dev.bq) do e
                    GC.@preserve src Mantle.storebytes!(e,dst.store,off,
                                                        Ptr{Cvoid}(pointer(src)),n)
                end
                expected = zeros(UInt8,n+8); expected[off+1:off+n] = src
                @test Array(dst) == expected
            finally
                Mantle.free!(dst)
            end
        end
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
end

function check_extent_refusal(ctx)
    dev = Mantle.lavadevice(ctx)
    be = Mantle.backend(dev)
    huge = 1 << (Sys.WORD_SIZE-2)
    @testset "invalid native shapes never submit or allocate" begin
        before = Mantle.submissions(dev)
        reserved = Mantle.reserved(Mantle.pool(dev))
        for (T,dims) in ((UInt8,(huge,4)),(Float64,(huge,)))
            @test_throws OverflowError Mantle.LavaArray{T}(undef,dims;bq=dev.bq)
            @test_throws OverflowError Mantle.KA.allocate(be,T,dims)
            @test Mantle.trydevicearray(dev,T,dims) === nothing
        end
        @test_throws ArgumentError Mantle.LavaArray{String}(undef,(0,);bq=dev.bq)
        @test_throws ArgumentError Mantle.LavaArray{Base.RefValue{Int}}(undef,(1,);bq=dev.bq)
        @test_throws ArgumentError Mantle.LavaArray{UInt8}(undef,(-1,-1);bq=dev.bq)
        @test_throws ArgumentError Mantle.KA.allocate(be,UInt8,(-1,-1))
        @test_throws ArgumentError Mantle.trydevicearray(dev,UInt8,(-1,-1))
        @test Mantle.submissions(dev) == before
        @test Mantle.reserved(Mantle.pool(dev)) == reserved
        @test isempty(Mantle.validationmessages!(dev))
    end
end

function check_copy_bounds(ctx)
    dev = Mantle.lavadevice(ctx)
    @testset "copy offsets respect logical views before native transfer" begin
        source = UInt32[7,8,9]
        bytes = collect(reinterpret(UInt8,source))
        words = reinterpret(UInt32,bytes)
        matrixwords = reinterpret(UInt32,reshape(bytes,4,3))
        spacedbytes = zeros(UInt8,2length(bytes)); spacedbytes[1:2:end] = bytes
        spacedwords = reinterpret(UInt32,view(spacedbytes,1:2:length(spacedbytes)))
        srcparent = Mantle.LavaArray(UInt32[10,20,30,40,50];bq=dev.bq)
        srcview = GPUArrays.derive(UInt32,srcparent,(3,),1)
        invalid = ((0,1,1),(4,1,1),(1,0,1),(1,4,1),(1,1,4),
                   (typemax(Int),1,2),(1,1,typemax(Int)))
        try
            for unified in (false,true)
                original = UInt32[101,102,103,104,105]
                parent = Mantle.LavaArray(original;bq=dev.bq,unified)
                dest = GPUArrays.derive(UInt32,parent,(3,),1)
                try
                    for src in (source,words,matrixwords,spacedwords,srcview)
                        before = Mantle.submissions(dev)
                        for (doff,soff,n) in invalid
                            @test_throws BoundsError copyto!(dest,doff,src,soff,n)
                        end
                        @test_throws ArgumentError copyto!(dest,1,src,1,-1)
                        @test copyto!(dest,typemax(Int),src,typemax(Int),0) === dest
                        @test Mantle.submissions(dev) == before
                        @test Array(parent) == original
                        @test copyto!(dest,Int32(2),src,UInt32(1),Int16(2)) === dest
                        expected = copy(original); expected[3:4] = Array(src)[1:2]
                        @test Array(parent) == expected
                        copyto!(parent,original)
                    end
                    before = Mantle.submissions(dev)
                    GC.@preserve source begin
                        pointer = Ptr{UInt8}(Base.pointer(source))
                        for direction in (:upload,:download), (off,n) in ((-1,4),(typemax(Int),4),(0,-1))
                            @test_throws ArgumentError Mantle.copy_buffer!(direction,parent.buf[],pointer,n;offset=off)
                        end
                    end
                    @test Mantle.submissions(dev) == before
                    @test Array(parent) == original
                finally
                    Mantle.unsafe_free!(dest); Mantle.unsafe_free!(parent)
                end
            end
            host = fill(UInt32(99),3)
            before = Mantle.submissions(dev)
            for (doff,soff,n) in invalid
                @test_throws BoundsError copyto!(host,doff,srcview,soff,n)
            end
            @test_throws ArgumentError copyto!(host,1,srcview,1,-1)
            @test copyto!(host,typemax(Int),srcview,typemax(Int),0) === host
            @test Mantle.submissions(dev) == before
            @test host == fill(UInt32(99),3)
            @test copyto!(host,Int32(1),srcview,UInt32(2),Int16(2)) === host
            @test host == UInt32[30,40,99]
        finally
            Mantle.unsafe_free!(srcview); Mantle.unsafe_free!(srcparent)
        end
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
end
