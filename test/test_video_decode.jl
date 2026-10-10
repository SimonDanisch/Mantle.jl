# Hardware H.264 decode — see `src/vulkan/runtime/video.jl`.
#
# Gated on `Mantle.videodecodes(dev)` and nothing else: the decoder is Vulkan
# Video, so a device created without a video-decode queue, and every Metal
# device, answers `false` and skips. The fixture is a 128x96 clip with
# multiple GOPs (-g 8) and B-pyramid (hierarchical B references), which specifically
# exercises the two paths that were real decoder bugs: GOP-aware display ordering
# (POC resets at each IDR) and MMCO reference-picture marking (dec_ref_pic_marking,
# used by B-pyramid to drop transient reference-B frames). Ground truth is the
# ffmpeg-decoded luma (Y) plane, checked in as raw bytes under `test/data/`.

using Test
using Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "H.264 hardware decode" begin
    dev = Mantle.Device(TESTBACKEND)
    # What a caller asks before opening a decode stream, on every handle it can
    # hold: the KA backend and the graph device say the same.
    @test Mantle.videodecodes(TESTBACKEND) === Mantle.videodecodes(dev)
    # A video-decode queue is necessary but not sufficient: the device must also
    # say whether the DPB and the decode target may share ONE image
    # (DPB_AND_OUTPUT_COINCIDE) or must be separate (DPB_AND_OUTPUT_DISTINCT).
    # The decoder implements both and picks per device; getting it wrong is
    # silent, since a distinct-only device (AMD Radeon 8060S reports flags=0x2)
    # cannot create a DST|DPB image at all and every frame then decodes to
    # all-zero — the comparison below failed with `243 == 0`, which reads like an
    # accuracy bug and is not one. Both layouts are checked against the same
    # ffmpeg ground truth, so this test means the same thing either way.
    if !Mantle.videodecodes(dev)
        @info "skipping H.264 decode test: this device has no hardware video decode"
        @test_skip Mantle.videodecodes(dev)
    else
        annexb = read(joinpath(@__DIR__, "data", "h264_decode_test.h264"))
        yref   = read(joinpath(@__DIR__, "data", "h264_decode_test_y.raw"))
        w, h = 128, 96
        nframes = length(yref) ÷ (w * h)

        # GPU-resident decode → device-local arrays, in DISPLAY order.
        gw, gh, frames = Mantle.decode_h264_gpu(annexb)
        @test (gw, gh) == (w, h)
        @test length(frames) == nframes
        # A device matrix, on the device under test: the frames never touch the
        # host until they are read below.
        @test frames[1] isa AbstractMatrix{UInt8}
        @test KA.get_backend(frames[1]) == TESTBACKEND

        # Pixel-exact vs ffmpeg across every frame (I, P, and hierarchical B).
        maxerr = 0
        for i in 1:nframes
            gi  = Array(frames[i])
            ref = reshape(yref[(i-1)*w*h + 1 : i*w*h], (w, h))
            maxerr = max(maxerr, maximum(abs.(Int.(gi) .- Int.(ref))))
        end
        @test maxerr == 0

        # Host path (downloads each frame) is consistent.
        hw, hh, host = Mantle.decode_h264_luma(annexb)
        @test (hw, hh) == (w, h)
        @test length(host) == nframes
        @test eltype(host[1]) == UInt8
        @test all(Array(frames[i]) == host[i] for i in 1:nframes)

        # Chunking must not change a single pixel. A decoder submits a chunk of
        # access units at a time, so the DPB crosses submit boundaries: one access
        # unit writes a slot the next reads as a reference, and frames are reordered
        # by (GOP, POC) afterwards. A boundary that dropped a reference, reused a
        # decode target too early or lost a barrier makes one-at-a-time and
        # all-at-once disagree.
        dec = Mantle.h264decoder(dev, annexb)
        got = Any[]
        try
            Mantle.feed!(dec, annexb)
            while Mantle.remaining(dec) > 0
                append!(got, Mantle.decodemore!(dec, 1))     # ONE access unit per call
            end
        finally
            close(dec)
        end
        @test length(got) == nframes
        @test all(Array(first(got[i])) == host[i] for i in 1:nframes)
    end
    # Where there is no decoder, asking for one says so instead of failing later.
    Mantle.videodecodes(dev) || @test_throws ArgumentError Mantle.h264decoder(dev, UInt8[])
end
