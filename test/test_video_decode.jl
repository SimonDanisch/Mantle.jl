# H.264 hierarchical B references and multiple GOPs, byte-exact against FFmpeg.

using Test
using Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "H.264 hardware decode" begin
    dev = Mantle.Device(TESTBACKEND)
    @test Mantle.videodecodes(TESTBACKEND) === Mantle.videodecodes(dev)
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

        # One-AU submissions must preserve references and display order across boundaries.
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
