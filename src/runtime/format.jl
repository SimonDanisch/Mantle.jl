# Pixel formats are element types, not an enum: `RGBA{Float16}` names
# `R16G16B16A16_SFLOAT` in a type kernels, `fill!` and readback can use.
# ColorTypes' layouts match Vulkan's byte for byte (`RGBA{N0f8}`/`BGRA{N0f8}`
# 8-bit UNORM, `RGBA{Float16}`/`RGBA{Float32}` SFLOAT, `Float32` as D32 depth,
# `UInt32` R32_UINT, `SVector{2,UInt32}` R32G32_UINT). sRGB changes conversion,
# not memory, so it is an image keyword; depth is the `Depth` usage.

"""Bytes one pixel occupies: `sizeof` of its element type, which is its layout."""
pixelbytes(::Type{T}) where {T} = sizeof(T)

# `vkformat` is declared in `src/vulkan/lowering.jl`, not here or in
# `BACKEND_VOCABULARY`: a name only one backend answers is that backend's
# function, like `mtlformat`.
