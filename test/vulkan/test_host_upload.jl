using Test, Mantle, GPUArrays, StaticArrays

function check_host_upload(ctx)
    @testset "typed uploads preserve shape order, bits and offsets" begin
        bq = ctx.default_bq
        dense = reshape(Float32.(1:6),2,3)
        sources = (dense, view(Float32.(1:12),2:2:12), PermutedDimsArray(dense,(2,1)),
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
