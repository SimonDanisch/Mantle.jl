"""
Pixel formats are element types, not an enum.

`Transient.Image(g, RGBA{Float16}, (w, h))` says the same thing as
`VK_FORMAT_R16G16B16A16_SFLOAT` and says it in a type a kernel can compute with,
`fill!` can write, and readback can return an array of. A parallel enum would be
a second name for something Julia already names, and the two would drift.

ColorTypes' layouts match Vulkan's byte for byte, which is what makes this work
rather than merely read well:

| element type    | bytes | Vulkan                          |
|-----------------|-------|---------------------------------|
| `RGBA{N0f8}`    | 4     | `R8G8B8A8_UNORM`                |
| `BGRA{N0f8}`    | 4     | `B8G8R8A8_UNORM`                |
| `RGBA{Float16}` | 8     | `R16G16B16A16_SFLOAT`           |
| `RGBA{Float32}` | 16    | `R32G32B32A32_SFLOAT`           |
| `Float32`       | 4     | `D32_SFLOAT` as a depth target  |

The one thing an element type cannot express is sRGB: `B8G8R8A8_SRGB` and
`B8G8R8A8_UNORM` have identical memory and differ only in how the hardware
converts on read and write. That is a property of the image, so it is a keyword
on the image and not part of the type.

Depth is not a separate format vocabulary either. `Float32` is the element type;
what makes an attachment a depth attachment is the `Depth` usage it is declared
with, which the sync vocabulary already carries.
"""

"""Bytes one pixel occupies. `sizeof` is the answer for every format whose
element type is its layout, which is all of them."""
pixelbytes(::Type{T}) where {T} = sizeof(T)

"""
    vkformat(backend, T; srgb = false) -> backend format

Implemented in the backend extension, which is the only place a format is named.
"""
function vkformat end
