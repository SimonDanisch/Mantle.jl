# Textures beyond what Metal.jl wraps: placing one in a heap, copies between a
# texture and buffer memory, mipmap generation, and reading texels back.

using Metal.ObjectiveC.Foundation: NSUInteger

"""
    heap_texture_size_and_align(dev, desc) -> MTLSizeAndAlign

How much heap space a texture matching `desc` needs, and what it must be aligned to.

Asked BEFORE the heap exists, which is the point: a placement heap has to be big
enough for everything that will live in it, and a texture cannot be created to find
out how big it is. The buffer side needs no equivalent because a buffer's size is
its byte count.
"""
heap_texture_size_and_align(dev::MTL.MTLDevice, desc::MTL.MTLTextureDescriptor) =
    @objc [dev::id{MTL.MTLDevice} heapTextureSizeAndAlignWithDescriptor:desc::id{MTL.MTLTextureDescriptor}]::MTL.MTLSizeAndAlign

"""
    heap_texture(heap, desc, offset) -> Union{MTLTexture,Nothing}

Place a texture described by `desc` at byte `offset` of a `MTLHeapTypePlacement`
heap, or `nothing` if Metal places none: the heap is too small, or `offset` is not a
multiple of the alignment [`heap_texture_size_and_align`](@ref) reported.

This is what lets a frame graph alias: two render targets whose lifetimes do not
overlap are created at the same offset and share the bytes. Not the same as a
texture over a buffer (`MTL.MTLTexture(buffer, desc, offset, bytesPerRow)`), which is
LINEAR, and an Apple GPU cannot render into a linear texture.

The texture retains its heap and is released when the returned wrapper is
collected: `newTextureWithDescriptor:offset:` returns a +1 reference, which the
managed wrapper adopts.
"""
heap_texture(heap::MTL.MTLHeap, desc::MTL.MTLTextureDescriptor, offset::Integer) =
    @objc [heap::id{MTL.MTLHeap} newTextureWithDescriptor:desc::id{MTL.MTLTextureDescriptor}
                                 offset:offset::NSUInteger]::Union{Nothing,MTL.MTLTexture}

"""
    getbytes!(dst::Ptr, tex, bytes_per_row, region, level = 0)

Copy a region of `tex` into host memory at `dst`, rows `bytes_per_row` apart.

Needed because a render target cannot be a buffer-backed (linear) texture on an
Apple GPU, so reading a rendered image back has to go through the texture rather
than through shared buffer memory. `tex` must have `Shared` or `Managed` storage,
and the work writing it must have completed: this is a host copy, ordered with
nothing on the GPU.
"""
function getbytes!(dst::Ptr, tex::MTL.MTLTexture, bytes_per_row::Integer,
                   region::MTL.MTLRegion, level::Integer = 0)
    @objc [tex::id{MTL.MTLTexture} getBytes:dst::Ptr{Cvoid}
                                   bytesPerRow:bytes_per_row::NSUInteger
                                   fromRegion:region::MTL.MTLRegion
                                   mipmapLevel:level::NSUInteger]::Nothing
end

# Copies between a texture and buffer memory, on a blit encoder. `MTL.append_copy!`
# copies buffer to buffer; these two take the texture side, `dst` first as there.

"""
    append_copy!(enc, dst::MTLBuffer, doff, bytes_per_row, bytes_per_image,
                 src::MTLTexture, origin, size, slice = 0, level = 0)

Copy a texture region into buffer memory.

The only way off a PRIVATE texture: `getbytes!` needs shared or managed storage, and
a render target the GPU writes wants neither. `bytes_per_image` may be 0 for a 2D
copy; Metal then derives it from the row stride and the height.
"""
function append_copy!(enc::MTL.MTLBlitCommandEncoderLike, dst::MTL.MTLBuffer, doff::Integer,
                      bytes_per_row::Integer, bytes_per_image::Integer,
                      src::MTL.MTLTexture, origin::MTL.MTLOrigin, size::MTL.MTLSize,
                      slice::Integer = 0, level::Integer = 0)
    @objc [enc::id{MTL.MTLBlitCommandEncoder} copyFromTexture:src::id{MTL.MTLTexture}
                                              sourceSlice:slice::NSUInteger
                                              sourceLevel:level::NSUInteger
                                              sourceOrigin:origin::MTL.MTLOrigin
                                              sourceSize:size::MTL.MTLSize
                                              toBuffer:dst::id{MTL.MTLBuffer}
                                              destinationOffset:doff::NSUInteger
                                              destinationBytesPerRow:bytes_per_row::NSUInteger
                                              destinationBytesPerImage:bytes_per_image::NSUInteger]::Nothing
end

"""
    append_copy!(enc, dst::MTLTexture, origin, size, src::MTLBuffer, soff,
                 bytes_per_row, bytes_per_image, slice = 0, level = 0)

Copy buffer memory into a texture region: the mirror of the method above.

The way ONTO a texture that is ordered with the rest of a queue. `replaceRegion`
writes from the CPU the moment it is called, while a command buffer that samples the
texture may still be running; a blit is encoded like any other command and waits
its turn. It is also the only way onto a PRIVATE texture, and the way to fill one
from data that is already on the device without a host round trip.
"""
function append_copy!(enc::MTL.MTLBlitCommandEncoderLike, dst::MTL.MTLTexture,
                      origin::MTL.MTLOrigin, size::MTL.MTLSize, src::MTL.MTLBuffer,
                      soff::Integer, bytes_per_row::Integer, bytes_per_image::Integer,
                      slice::Integer = 0, level::Integer = 0)
    @objc [enc::id{MTL.MTLBlitCommandEncoder} copyFromBuffer:src::id{MTL.MTLBuffer}
                                              sourceOffset:soff::NSUInteger
                                              sourceBytesPerRow:bytes_per_row::NSUInteger
                                              sourceBytesPerImage:bytes_per_image::NSUInteger
                                              sourceSize:size::MTL.MTLSize
                                              toTexture:dst::id{MTL.MTLTexture}
                                              destinationSlice:slice::NSUInteger
                                              destinationLevel:level::NSUInteger
                                              destinationOrigin:origin::MTL.MTLOrigin]::Nothing
end

"""
    append_generate_mipmaps!(enc, tex)

Fill the mip levels of `tex` above level 0 from level 0, each a filtered half of the
one before. `tex` must have been made with more than one level.
"""
append_generate_mipmaps!(enc::MTL.MTLBlitCommandEncoderLike, tex::MTL.MTLTexture) =
    @objc [enc::id{MTL.MTLBlitCommandEncoder} generateMipmapsForTexture:tex::id{MTL.MTLTexture}]::Nothing
