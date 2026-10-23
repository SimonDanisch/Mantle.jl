# Video-decode entry points that default to `vk_context()`.

"""
    decode_h264_gpu(annexb; ctx = vk_context(), maxframes = typemax(Int)) -> (width, height, ys)

Hardware-decode an H.264 Annex-B stream. `ys` holds each frame's luma plane as a
device-local `LavaArray{UInt8,2}` cropped to the display size, in display order.
Requires `ctx.video_decode_available`.
"""
function decode_h264_gpu(annexb::Vector{UInt8}; ctx::LavaDevice = vk_context(), kw...)
    w, h, ys, _ = VideoDecode.decode_h264(ctx, annexb; kw...)
    return (w, h, ys)
end

"""
    decode_h264_nv12(annexb; ctx = vk_context(), maxframes = typemax(Int)) -> (width, height, ys, uvs)

Like [`decode_h264_gpu`](@ref), plus `uvs`: each frame's interleaved U/V plane as
a `width × (height ÷ 2)` `LavaArray{UInt8,2}`. GPUFiltering's `nv12torgb!`
converts the pair to RGB.
"""
function decode_h264_nv12(annexb::Vector{UInt8}; ctx::LavaDevice = vk_context(), kw...)
    w, h, ys, uvs = VideoDecode.decode_h264(ctx, annexb; chroma=true, kw...)
    return (w, h, ys, uvs)
end

# `decode_h264_luma` (this decode plus a download) is defined in core. The
# `LavaDevice` method of `h264decoder` is in `graph.jl`.

