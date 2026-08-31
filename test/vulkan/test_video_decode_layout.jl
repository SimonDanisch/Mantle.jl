# Decoder invariants that the end-to-end decode test cannot pin on one machine.
#
# test_video_decode.jl checks decoded pixels against ffmpeg, which only exercises
# whichever DPB layout the running device happens to offer. These are the two
# properties that were silently wrong and that CAN be asserted on any device.
#
# 1. Bitstream alignment came from a hardcoded 256. The H.264 profile on AMD asks
#    for 4096, and a srcBufferOffset/srcBufferRange that is not a multiple of the
#    profile's requirement is a decode that produces nothing at all — no error, no
#    validation message without layers, just all-zero frames. Anything that
#    reintroduces a constant here fails on any device asking for more than it.
#
# 2. A DISTINCT-only device must actually get separate decode targets. If that
#    silently fell back to the coincide allocation, `vkCreateImage` would reject
#    VIDEO_DECODE_DST|VIDEO_DECODE_DPB and every frame would decode to zero.

using Test, Lava, Mantle
const VD = MVE.VideoDecode

@testset "H.264 decoder layout + bitstream alignment" begin
    ctx = MVE.vk_context()

    if !ctx.video_decode_available || !VD.decode_supported(ctx)
        @info "skipping decoder layout test: no usable video-decode support on $(ctx.device_name)"
        @test_skip ctx.video_decode_available
    else
        annexb = read(joinpath(@__DIR__, "data", "h264_decode_test.h264"))
        dec = VD.H264Decoder(ctx, annexb)
        try
            # The profile's own requirements, asked for independently of the decoder.
            caps = VD.decode_capability_flags(ctx)
            @test (caps & 0x3) != 0

            # Whatever alignment the decoder settled on must satisfy BOTH the offset
            # and the size requirement, and must not be a constant that ignores them.
            @test dec.bsalign > 0
            @test ispow2(dec.bsalign)

            # Every bitstream offset the decoder hands to vkCmdDecodeVideoKHR is a
            # multiple of bsalign (it advances by `cld(off, bsalign) * bsalign`), so
            # bsalign being a multiple of the requirement is the whole invariant.
            props = VD.video_capabilities(ctx)
            # A device reporting 0 would mean "no alignment requirement"; guard so a
            # zero never turns this assertion into a DivideError.
            for req in (props.minBitstreamBufferOffsetAlignment,
                        props.minBitstreamBufferSizeAlignment)
                @test req >= 0
                req > 0 && @test dec.bsalign % req == 0
            end

            # A DISTINCT-only device must take the separate-target path; a COINCIDE
            # device must not pay for it.
            if VD.decode_coincide_supported(ctx)
                @test dec.mkoutimg === nothing
                @test isempty(dec.outimgs)
            else
                @test dec.mkoutimg !== nothing
            end
        finally
            close(dec)
        end

        # Chunking must not change a single pixel.
        #
        # The decoder submits a chunk of access units per video-queue command
        # buffer, and under DISTINCT it further splits that into sub-chunks of
        # MAX_DECODE_TARGETS so the decode targets can be reused. Both make the
        # DPB cross a submit boundary: this AU writes a slot the next one reads as
        # a reference, and the frames are reordered by (GOP, POC) afterwards. If a
        # boundary dropped a reference, reused a target too early, or lost the
        # inter-decode barrier, one-AU-at-a-time and all-at-once would disagree.
        yref = read(joinpath(@__DIR__, "data", "h264_decode_test_y.raw"))
        w, h = 128, 96
        nframes = length(yref) ÷ (w * h)

        _, _, batched = MVE.decode_h264_gpu(annexb)          # one big chunk
        @test length(batched) == nframes

        incremental = MVE.VideoDecode.H264Decoder(MVE.vk_context(), annexb)
        got = Any[]
        try
            MVE.VideoDecode.feed!(incremental, annexb)
            while MVE.VideoDecode.remaining(incremental) > 0
                append!(got, MVE.VideoDecode.decodemore!(incremental, 1))   # ONE AU per submit
            end
        finally
            close(incremental)
        end
        @test length(got) == nframes

        # Both paths, pixel-identical, and both matching ffmpeg.
        for i in 1:nframes
            ref = reshape(yref[(i-1)*w*h + 1 : i*w*h], (w, h))
            @test Array(batched[i]) == ref
            @test Array(got[i][1])  == ref
        end
    end
end
