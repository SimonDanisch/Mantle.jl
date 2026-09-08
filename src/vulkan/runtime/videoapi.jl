# The user-facing video-decode entry points.
#
# Wrappers over `VideoDecode.decode_h264` that supply `vk_context()` — so a
# caller does not hold a context — which is exactly why they could not stay in
# Lava when it stopped having one. They were the tail of `Lava.jl`; the decoder
# they call has been beside them in `runtime/video.jl` all along.

"""
    decode_h264_gpu(annexb::Vector{UInt8}; maxframes) -> (width, height, Vector{LavaArray{UInt8,2}})

Hardware-decode an H.264 Annex-B elementary stream on the GPU (VK_KHR_video_decode),
keeping every decoded luma (Y) plane GPU-resident: each frame is returned as a
device-local `LavaArray{UInt8,2}` (cropped to the display size), never touching host
memory. Feed these straight into Lava kernels / the motion tracker. Requires a device
created with video decode support (`vk_context().video_decode_available`).
"""
function decode_h264_gpu(annexb::Vector{UInt8}; ctx::VkContext = vk_context(), kw...)
    w, h, ys, _ = VideoDecode.decode_h264(ctx, annexb; kw...)
    return (w, h, ys)
end

"""
    decode_h264_nv12(annexb; maxframes) -> (width, height, ys, uvs)

Like [`decode_h264_gpu`] but also returns the chroma plane: `ys` are the luma
`LavaArray{UInt8,2}` (width×height) and `uvs` the interleaved-U/V chroma
`LavaArray{UInt8,2}` (width×(height÷2)) for each frame — the two planes of NV12,
GPU-resident. Convert to RGB on-GPU with GPUFiltering's `nv12torgb!`.
"""
function decode_h264_nv12(annexb::Vector{UInt8}; ctx::VkContext = vk_context(), kw...)
    w, h, ys, uvs = VideoDecode.decode_h264(ctx, annexb; chroma=true, kw...)
    return (w, h, ys, uvs)
end

"""
    decode_h264_luma(annexb::Vector{UInt8}; maxframes) -> (width, height, Vector{Matrix{UInt8}})

Like [`decode_h264_gpu`] but downloads each decoded luma plane to a host
`Matrix{UInt8}` (grayscale = the NV12 Y plane). Used to validate the GPU decoder
against a reference (e.g. ffmpeg).
"""
function decode_h264_luma(annexb::Vector{UInt8}; kw...)
    w, h, frames = decode_h264_gpu(annexb; kw...)
    return (w, h, Matrix{UInt8}[Array(f) for f in frames])
end

# ---- Profiling (kernel SPIR-V stats + per-dispatch GPU timing) ----

# ---- Phase 2: Graphics ----

# ---- Phase 2: Ray Tracing ----

# ---- Compute kernels ----

