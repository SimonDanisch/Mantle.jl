using Test, Mantle

function check_video_session(ctx, D, file; chroma=true)
    V = Mantle.VideoDecode
    bytes = read(joinpath(@__DIR__,"..","data",file))
    reference = read(joinpath(@__DIR__,"..","data",replace(file,r"\.[^.]+$"=>".nv12")))
    dec = D(ctx,bytes;chroma)
    try
        Mantle.feed!(dec,bytes)
        count = length(dec.PIN)
        memory = dec.bmem
        V.ensurebitbuf!(dec,dec.bufsz+4096)
        @test dec.bmem !== memory
        @test length(dec.PIN) == count
        frames = Any[]
        while Mantle.remaining(dec)>0
            append!(frames,Mantle.decodemore!(dec,3))
        end
        @test length(frames) == 24
        got = reduce(vcat,(chroma ? vcat(vec(Array(y)),vec(Array(uv))) : vec(Array(y)) for (y,uv) in frames))
        if chroma
            @test got == reference
        else
            stride = dec.DW*dec.DH*3÷2
            yref = reduce(vcat,(reference[i:i+dec.DW*dec.DH-1] for i in 1:stride:length(reference)))
            @test got == yref
        end
    finally
        close(dec)
    end
    @test !dec.open && Mantle.remaining(dec)==0
    @test dec.bmem === dec.bbuf === dec.cbh === nothing
    @test isempty(dec.PIN) && isempty(dec.imgs) && isempty(dec.outimgs)
    @test_throws ArgumentError Mantle.feed!(dec,bytes)
    @test_throws ArgumentError Mantle.decodemore!(dec,1)
    @test close(dec) === nothing
    dims, expected = (192,192), 5
    w,h,ys,uvs = V.decodevideo(D,ctx,bytes;maxframes=expected,chroma)
    @test (w,h) == dims
    @test length(ys) == expected
    @test length(uvs) == (chroma ? expected : 0)
    _,_,ys,uvs = V.decodevideo(D,ctx,bytes;maxframes=0,chroma)
    @test isempty(ys) && isempty(uvs)
    @test_throws ArgumentError V.decodevideo(D,ctx,bytes;maxframes=-1)
end

function check_video_sessions(ctx)
    V = Mantle.VideoDecode
    @testset "shared video session ownership and decoding" begin
        if !ctx.video_decode_available
            @test_skip ctx.video_decode_available
            return
        end
        check_video_session(ctx,V.H264Decoder,"h264_type2_test.h264")
        if Mantle.has_extension(ctx.physical_device,"VK_KHR_video_decode_h265")
            caps = V.videocaps(ctx,V.decoderprofile(V.H265Decoder),V.codeccapabilities(V.H265Decoder))
            if caps.flags & UInt32(1) != 0
                check_video_session(ctx,V.H265Decoder,"h265_decode_test.h265")
                check_video_session(ctx,V.H265Decoder,"h265_decode_test.h265";chroma=false)
            else
                @test_throws ErrorException V.H265Decoder(ctx,read(joinpath(@__DIR__,"..","data","h265_decode_test.h265")))
            end
        end
        # Preferences may not override driver eligibility or required flags.
        localbit = Mantle.VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        for (i,mt) in enumerate(ctx.memory_properties.memory_types)
            mask = UInt32(1) << (i-1)
            @test Mantle.find_memory_type(ctx,mask,Mantle.VK.MemoryPropertyFlag(0);preferred_flags=localbit)==i-1
            if mt.property_flags & localbit != localbit
                @test_throws Mantle.LavaError Mantle.find_memory_type(ctx,mask,localbit)
            end
        end
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
