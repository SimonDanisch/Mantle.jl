# Backend capability/refusal vocabulary available on every platform.

"""staged_gemm_tile() -> Union{Int,Nothing}

Square cooperative-matrix extent supported by this build's staged GEMM, or
nothing. Eligibility also requires caps(device).coopmat and a matching device tile."""
function staged_gemm_tile end

"""videodecodes(dev) -> Bool

Whether this device/backend exposes a hardware video-decode session. Currently
only video-enabled Vulkan devices implement it; other devices return false."""
videodecodes(dev) = false

"""decode_h264_gpu(annexb::AbstractVector{UInt8}; maxframes) -> (width, height, frames)

Decode Annex-B H.264 into display-sized GPU luma planes. Requires videodecodes."""
decode_h264_gpu(annexb::AbstractVector{UInt8}; kw...) = throw(ArgumentError(
    "decode_h264_gpu: this device has no hardware video decode; ask `videodecodes(device)` first"))

"""
    decode_h264_luma(annexb; maxframes) -> (width, height, Vector{Matrix{UInt8}})

[`decode_h264_gpu`](@ref), with every decoded luma plane downloaded to the host. Used
to check the decoder against a reference such as ffmpeg.
"""
function decode_h264_luma(annexb::AbstractVector{UInt8}; kw...)
    w, h, frames = decode_h264_gpu(annexb; kw...)
    return (w, h, Matrix{UInt8}[Array(f) for f in frames])
end

"""h264decoder(dev, paramnals::AbstractVector{UInt8}; chroma=false) -> decoder

Open a hardware session from SPS/PPS NALs. Feed chunks beginning at an IDR,
use decodemore! to receive display-order frames, and close when finished.
Requires videodecodes(dev)."""
h264decoder(dev, paramnals::AbstractVector{UInt8}; kw...) = throw(ArgumentError(
    "h264decoder: this device has no hardware video decode; ask `videodecodes(device)` first"))

"""
    feed!(decoder, annexb) -> decoder

Queue an Annex-B chunk, typically one GOP starting at an IDR, on a decoder from
[`h264decoder`](@ref). Frames not yet returned from an earlier chunk come out first.
"""
function feed! end

"""decodemore!(decoder, n) -> Vector{(Y, UV)}

Decode up to n queued access units. Return display-ready frames in order;
reordered frames may be held until the chunk ends."""
function decodemore! end

"""
    remaining(decoder) -> Int

How many frames of what was fed [`decodemore!`](@ref) has not returned yet.
"""
function remaining end

"""supports_float64(dev) -> Bool

Whether the device/backend supports Float64 and ComplexF64 arrays and kernels."""
supports_float64(dev) = false

"""supports_int64_atomics(dev) -> Bool

Whether device-memory Int64/UInt64 atomic updates are supported. The default is
false; backends answer from their enabled native capabilities."""
supports_int64_atomics(dev) = false

"""isoutofmemory(err) -> Bool

Recognize an allocation refusal that core may collect/trim/retry. Other errors
propagate. Backends extend Julia's OutOfMemoryError with native OOM codes."""
isoutofmemory(err) = err isa OutOfMemoryError
