# What core asks a backend, outside the graph.
#
# `graph/backend.jl` holds the graph's own list and says why it is short. These
# two are not graph questions — one is the compiled-kernel cache, the other is
# which matrix kernels this build has — and they are here for the same reason
# that list exists: `src/vulkan/` and `src/metal/` are `@static include`d, so a
# name defined under one of them does not EXIST under the other. It is not an
# empty table or a disabled feature; there is no binding.
#
# Same shape as `initbackend!`, deliberately: **declared here, answered in
# `src/vulkan/` or `src/metal/`.** A core file that instead wrote the answer
# would have to name the backend's types to write it, which is what
# `test_core_names_no_backend.jl` catches.
#
# ── The alternative, and why it is not this ───────────────────────────────────
#
# A caller can reach into this module by name instead — `isdefined(Mantle,
# :GEMM_TILINGS)` — and that is what `DNNKernels.coopmatkernels` and fifteen
# runners' `__init__` did. It works. It also keeps working through a rename, a
# split of the table, or a second backend that implements the same feature at a
# different width and is then silently admitted, because the consumer is
# spelling the provider's internal names and nobody agreed to that contract.
#
# A backend that forgets one of these gets a `MethodError` naming the function
# it did not answer, which is the failure worth having: the `isdefined` version
# fails instead at `Mantle.GEMM_TILINGS`, two frames into a kernel, as an
# `UndefVarError` on a name the caller was never supposed to know.


"""
    staged_gemm_tile() -> Union{Int,Nothing}

The square cooperative-matrix extent this BUILD's staged GEMM is emitted at, or
`nothing` if this build has no staged GEMM.

A caller that wants those kernels needs two answers and this is the second. The
first is about the machine and `KernelInterface.DeviceCaps` gives it: `coopmat`
says the driver has cooperative matrices, `tile` says how wide its square is.
Neither says whether MANTLE has kernels for that width, and neither can — a
`DeviceCaps` is a record of a device and carries no backend identity, which is
what makes it portable and what stops this being a dispatch. So both halves, and
the tile makes them one comparison:

    dev.coopmat && (t = Mantle.staged_gemm_tile()) !== nothing && dev.tile == t

Answering with the tile rather than with a `Bool` is the point of the `Int`: the
caller had grown its own copy of `16` to compare `dev.tile` against, and a
restated constant is a second place for the two to disagree.
"""
function staged_gemm_tile end

"""
    videodecodes(dev) -> Bool

Whether `dev` has a hardware video decode session that `VideoDecode` drives.

What a caller asks before it opens a GPU decode stream. Only the Vulkan backend has
a decoder, and only on a device created with video decode; every other device
answers `false` here. Before this was declared the editor asked
`Mantle.vk_context().video_decode_available`, which on a build without the Vulkan
backend is an `UndefVarError`. Its export path caught that and fell back to the
CPU reader with a warning on every source, so on Metal the lane was chosen by an
exception instead of by the device.

`dev` is whatever the caller launches on: a device, or the KernelAbstractions
backend that stands for one.
"""
videodecodes(dev) = false

"""
    decode_h264_gpu(annexb::AbstractVector{UInt8}; maxframes) -> (width, height, frames)

Hardware-decode an H.264 Annex-B elementary stream, keeping every decoded luma (Y)
plane on the device: each frame is a device array of `UInt8`, cropped to the display
size. Only on a device that [`videodecodes`](@ref); everywhere else this refuses.
"""
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

"""
    supports_float64(dev) -> Bool

Whether kernels and arrays on `dev` can hold `Float64` (and `ComplexF64`).

Apple GPUs have no double precision and Metal refuses to allocate a `Float64`
array, so a caller choosing an element type, or a test looping over element
types, asks this instead of naming a backend. Every Vulkan device Mantle builds
enables `shaderFloat64` at creation, so the Vulkan backend answers `true`.

`dev` is a device or the KernelAbstractions backend that stands for one.
"""
supports_float64(dev) = false

"""
    supports_int64_atomics(dev) -> Bool

Whether a kernel on `dev` can update a 64-bit integer in device memory atomically:
`Atomix.@atomic` add, sub, and the rest on an `Int64` or `UInt64` element.

Apple GPUs cannot; Metal has 64-bit `umin` and `umax` on device memory and only
without the result, so a kernel that adds to an `Int64` fails to compile there. A
Vulkan device answers what its `shaderBufferInt64Atomics` feature says.

`dev` is a device or the KernelAbstractions backend that stands for one.
"""
supports_int64_atomics(dev) = false
