# Hardware H.264 decode (VK_KHR_video_decode) — see runtime/video.jl.
#
# Skips on devices without a video-decode queue. The fixture is a 128x96 clip with
# multiple GOPs (-g 8) and B-pyramid (hierarchical B references), which specifically
# exercises the two paths that were real decoder bugs: GOP-aware display ordering
# (POC resets at each IDR) and MMCO reference-picture marking (dec_ref_pic_marking,
# used by B-pyramid to drop transient reference-B frames). Ground truth is the
# ffmpeg-decoded luma (Y) plane, checked in as raw bytes.

using Test
using Lava, Mantle
@testset "H.264 hardware decode" begin
    ctx = MVE.vk_context()
    # A video-decode queue is necessary but not sufficient: the device must also
    # say whether the DPB and the decode target may share ONE image
    # (DPB_AND_OUTPUT_COINCIDE) or must be separate (DPB_AND_OUTPUT_DISTINCT).
    # The decoder implements both and picks per device; getting it wrong is
    # silent, since a distinct-only device (AMD Radeon 8060S reports flags=0x2)
    # cannot create a DST|DPB image at all and every frame then decodes to
    # all-zero — the comparison below failed with `243 == 0`, which reads like an
    # accuracy bug and is not one. Both layouts are checked against the same
    # ffmpeg ground truth, so this test means the same thing either way.
    if !ctx.video_decode_available
        @info "skipping H.264 decode test: no video-decode queue on $(ctx.device_name)"
        @test_skip ctx.video_decode_available
    elseif !MVE.VideoDecode.decode_supported(ctx)
        @info "skipping H.264 decode test: $(ctx.device_name) offers neither " *
              "DPB_AND_OUTPUT_COINCIDE nor DPB_AND_OUTPUT_DISTINCT " *
              "(flags=0x$(string(MVE.VideoDecode.decode_capability_flags(ctx), base=16)))"
        @test_skip MVE.VideoDecode.decode_supported(ctx)
    else
        @info "H.264 decode on $(ctx.device_name): " *
              (MVE.VideoDecode.decode_coincide_supported(ctx) ? "COINCIDE" : "DISTINCT") *
              " layout (flags=0x$(string(MVE.VideoDecode.decode_capability_flags(ctx), base=16)))"
        annexb = read(joinpath(@__DIR__, "data", "h264_decode_test.h264"))
        yref   = read(joinpath(@__DIR__, "data", "h264_decode_test_y.raw"))
        w, h = 128, 96
        nframes = length(yref) ÷ (w * h)

        # GPU-resident decode → device-local LavaArrays, in DISPLAY order.
        gw, gh, frames = MVE.decode_h264_gpu(annexb)
        @test (gw, gh) == (w, h)
        @test length(frames) == nframes
        @test frames[1] isa MVE.LavaArray{UInt8,2}

        # Pixel-exact vs ffmpeg across every frame (I, P, and hierarchical B).
        maxerr = 0
        for i in 1:nframes
            gi  = Array(frames[i])
            ref = reshape(yref[(i-1)*w*h + 1 : i*w*h], (w, h))
            maxerr = max(maxerr, maximum(abs.(Int.(gi) .- Int.(ref))))
        end
        @test maxerr == 0

        # Host path (downloads each frame via Array(::VideoImage)) is consistent.
        hw, hh, host = MVE.decode_h264_luma(annexb)
        @test (hw, hh) == (w, h)
        @test length(host) == nframes
        @test eltype(host[1]) == UInt8
        @test all(Array(frames[i]) == host[i] for i in 1:nframes)
    end
end
