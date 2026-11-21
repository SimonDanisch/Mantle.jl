# Render pipelines, render passes and draws: the rasterisation half of the command
# interface, which Metal.jl does not wrap (it compiles and dispatches compute only).
# The ObjC classes and their properties exist in Metal.jl's `libmtl.jl`; this is the
# Julia layer over them. Creating an object of a Metal.jl type is a Mantle function
# here (`render_pipeline_descriptor()`), not a constructor of that type.
#
# Slot indices are 1-based throughout, like Metal.jl's compute encoder: Julia counts
# buffer slots from one, Metal from zero, and the conversion belongs at the boundary
# rather than in every caller.

using Metal.ObjectiveC: Out
using Metal.ObjectiveC.Foundation: NSError, NSUInteger

# ── Descriptors and states ──────────────────────────────────────────────────

"""A new, empty render pass descriptor."""
render_pass_descriptor() =
    @objc [MTLRenderPassDescriptor renderPassDescriptor]::MTL.MTLRenderPassDescriptor

"""A new render pipeline descriptor, for a vertex and a fragment function."""
render_pipeline_descriptor() =
    @objc [MTLRenderPipelineDescriptor new]::MTL.MTLRenderPipelineDescriptor

"""
A new mesh render pipeline descriptor. A mesh pipeline is created from its OWN
descriptor type, not from `MTLRenderPipelineDescriptor` with a mesh function set on
it: the object and mesh stages have threadgroup sizes and a payload length that a
vertex pipeline has nowhere to put.
"""
mesh_render_pipeline_descriptor() =
    @objc [MTLMeshRenderPipelineDescriptor new]::MTL.MTLMeshRenderPipelineDescriptor

"""A new depth/stencil descriptor."""
depth_stencil_descriptor() =
    @objc [MTLDepthStencilDescriptor new]::MTL.MTLDepthStencilDescriptor

"""A new sampler descriptor."""
sampler_descriptor() = @objc [MTLSamplerDescriptor new]::MTL.MTLSamplerDescriptor

"""
    render_pipeline_state(dev, desc) -> MTLRenderPipelineState

Compile the pipeline `desc` describes, a vertex or a mesh pipeline, for `dev`.
Throws the `NSError` Metal reports when it cannot.
"""
function render_pipeline_state(dev::MTL.MTLDevice, desc::MTL.MTLRenderPipelineDescriptor)
    err = Out{NSError}()
    state = @objc [dev::id{MTL.MTLDevice} newRenderPipelineStateWithDescriptor:desc::id{MTL.MTLRenderPipelineDescriptor}
                   error:err::Out{NSError}]::Union{Nothing,MTL.MTLRenderPipelineState}
    state === nothing &&
        throw(something(err[], ErrorException("Metal created no render pipeline state")))
    return state
end

function render_pipeline_state(dev::MTL.MTLDevice, desc::MTL.MTLMeshRenderPipelineDescriptor)
    err = Out{NSError}()
    state = @objc [dev::id{MTL.MTLDevice} newRenderPipelineStateWithMeshDescriptor:desc::id{MTL.MTLMeshRenderPipelineDescriptor}
                   options:MTL.MTLPipelineOptionNone::MTL.MTLPipelineOption
                   reflection:C_NULL::Ptr{Cvoid}
                   error:err::Out{NSError}]::Union{Nothing,MTL.MTLRenderPipelineState}
    state === nothing &&
        throw(something(err[], ErrorException("Metal created no mesh render pipeline state")))
    return state
end

"""The depth/stencil state `desc` describes, for `dev`."""
depth_stencil_state(dev::MTL.MTLDevice, desc::MTL.MTLDepthStencilDescriptor) =
    @objc [dev::id{MTL.MTLDevice} newDepthStencilStateWithDescriptor:desc::id{MTL.MTLDepthStencilDescriptor}]::MTL.MTLDepthStencilState

"""The sampler `desc` describes, for `dev`."""
sampler_state(dev::MTL.MTLDevice, desc::MTL.MTLSamplerDescriptor) =
    @objc [dev::id{MTL.MTLDevice} newSamplerStateWithDescriptor:desc::id{MTL.MTLSamplerDescriptor}]::MTL.MTLSamplerState

"""
    colorattachment(desc, i)

The `i`-th (1-based) colour attachment of a pipeline or a pass descriptor.

`desc.colorAttachments` is an ObjC array-like object addressed with
`objectAtIndexedSubscript:`, not an NSArray, so nothing generic reaches it. Both the
pipeline's and the pass's arrays need it, and without it there is no way to name a
colour target at all.
"""
function colorattachment(desc::Union{MTL.MTLRenderPipelineDescriptor,
                                     MTL.MTLMeshRenderPipelineDescriptor}, i::Integer)
    arr = desc.colorAttachments
    @objc [arr::id{MTL.MTLRenderPipelineColorAttachmentDescriptorArray} objectAtIndexedSubscript:(i - 1)::NSUInteger]::MTL.MTLRenderPipelineColorAttachmentDescriptor
end

function colorattachment(desc::MTL.MTLRenderPassDescriptor, i::Integer)
    arr = desc.colorAttachments
    @objc [arr::id{MTL.MTLRenderPassColorAttachmentDescriptorArray} objectAtIndexedSubscript:(i - 1)::NSUInteger]::MTL.MTLRenderPassColorAttachmentDescriptor
end

# ── The encoder ─────────────────────────────────────────────────────────────

"""A render command encoder for the pass `desc` on `cmdbuf`; end it with `MTL.endEncoding!`."""
render_command_encoder(cmdbuf::MTL.MTLCommandBuffer, desc::MTL.MTLRenderPassDescriptor) =
    @objc [cmdbuf::id{MTL.MTLCommandBuffer} renderCommandEncoderWithDescriptor:desc::id{MTL.MTLRenderPassDescriptor}]::MTL.MTLRenderCommandEncoder

set_pipeline!(rce::MTL.MTLRenderCommandEncoderLike, pip::MTL.MTLRenderPipelineState) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setRenderPipelineState:pip::id{MTL.MTLRenderPipelineState}]::Nothing

set_depth_stencil_state!(rce::MTL.MTLRenderCommandEncoderLike, st::MTL.MTLDepthStencilState) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setDepthStencilState:st::id{MTL.MTLDepthStencilState}]::Nothing

set_cull_mode!(rce::MTL.MTLRenderCommandEncoderLike, mode::MTL.MTLCullMode) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setCullMode:mode::MTL.MTLCullMode]::Nothing

"""
    set_front_facing_winding!(rce, winding)

Which triangle winding counts as front-facing.

Needed by anything that flips a clip axis: mirroring y reverses the handedness of
every triangle, so what was counter-clockwise on screen becomes clockwise and
back-face culling starts keeping exactly the faces it used to drop.
"""
set_front_facing_winding!(rce::MTL.MTLRenderCommandEncoderLike, w::MTL.MTLWinding) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setFrontFacingWinding:w::MTL.MTLWinding]::Nothing

# Inline argument bytes, one family per stage: the selectors are distinct and Metal
# binds nothing if the wrong one is used. Metal copies the bytes before returning.

set_vertex_bytes!(rce::MTL.MTLRenderCommandEncoderLike, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setVertexBytes:ptr::Ptr{Cvoid}
                                                length:len::NSUInteger
                                                atIndex:(index - 1)::NSUInteger]::Nothing

set_fragment_bytes!(rce::MTL.MTLRenderCommandEncoderLike, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setFragmentBytes:ptr::Ptr{Cvoid}
                                                length:len::NSUInteger
                                                atIndex:(index - 1)::NSUInteger]::Nothing

set_mesh_bytes!(rce::MTL.MTLRenderCommandEncoderLike, ptr::Ptr, len::Integer, index::Integer) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setMeshBytes:ptr::Ptr{Cvoid}
                                                length:len::NSUInteger
                                                atIndex:(index - 1)::NSUInteger]::Nothing

# A sampled texture needs a texture and a sampler, and on Metal they occupy two
# separate slot namespaces rather than one combined descriptor.

set_fragment_texture!(rce::MTL.MTLRenderCommandEncoderLike, tex::MTL.MTLTexture, index::Integer) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setFragmentTexture:tex::id{MTL.MTLTexture}
                                                atIndex:(index - 1)::NSUInteger]::Nothing

set_fragment_sampler!(rce::MTL.MTLRenderCommandEncoderLike, st::MTL.MTLSamplerState, index::Integer) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setFragmentSamplerState:st::id{MTL.MTLSamplerState}
                                                atIndex:(index - 1)::NSUInteger]::Nothing

set_viewport!(rce::MTL.MTLRenderCommandEncoderLike, vp::MTL.MTLViewport) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setViewport:vp::MTL.MTLViewport]::Nothing

set_scissor!(rce::MTL.MTLRenderCommandEncoderLike, r::MTL.MTLScissorRect) =
    @objc [rce::id{MTL.MTLRenderCommandEncoder} setScissorRect:r::MTL.MTLScissorRect]::Nothing

"""
    use!(rce, buf, mode = MTL.ReadUsage, stages = vertex | fragment)

Make `buf` resident for this render pass.

Needed whenever a shader reaches memory through an ADDRESS rather than through a
bound slot, which is what an `MtlDeviceArray` argument is: a struct holding a
`gpuAddress`. Metal only maps what an encoder was told about, so an address the
encoder never saw reads as zeros: silently, and only when the page happens not to
be mapped. Metal.jl's launch does this for compute encoders (`MTL.use!`); a render
encoder has no such hook, so the draw path calls this itself.
"""
function use!(rce::MTL.MTLRenderCommandEncoderLike, buf::MTL.MTLBuffer,
              mode::MTL.MTLResourceUsage = MTL.ReadUsage,
              stages::MTL.MTLRenderStages = MTL.MTLRenderStages(MTL.MTLRenderStageVertex |
                                                                MTL.MTLRenderStageFragment))
    @objc [rce::id{MTL.MTLRenderCommandEncoder} useResource:buf::id{MTL.MTLBuffer}
                                                usage:mode::MTL.MTLResourceUsage
                                                stages:stages::MTL.MTLRenderStages]::Nothing
end

# ── Draws ───────────────────────────────────────────────────────────────────

function draw_primitives!(rce::MTL.MTLRenderCommandEncoderLike, prim::MTL.MTLPrimitiveType,
                          first::Integer, count::Integer, instances::Integer = 1)
    @objc [rce::id{MTL.MTLRenderCommandEncoder} drawPrimitives:prim::MTL.MTLPrimitiveType
                                                vertexStart:first::NSUInteger
                                                vertexCount:count::NSUInteger
                                                instanceCount:instances::NSUInteger]::Nothing
end

function draw_indexed_primitives!(rce::MTL.MTLRenderCommandEncoderLike, prim::MTL.MTLPrimitiveType,
                                  count::Integer, itype::MTL.MTLIndexType,
                                  ibuf::MTL.MTLBuffer, ioffset::Integer,
                                  instances::Integer = 1)
    @objc [rce::id{MTL.MTLRenderCommandEncoder} drawIndexedPrimitives:prim::MTL.MTLPrimitiveType
                                                indexCount:count::NSUInteger
                                                indexType:itype::MTL.MTLIndexType
                                                indexBuffer:ibuf::id{MTL.MTLBuffer}
                                                indexBufferOffset:ioffset::NSUInteger
                                                instanceCount:instances::NSUInteger]::Nothing
end

"""
    draw_primitives_indirect!(rce, prim, buf, offset = 0)

The counterpart to a compute indirect dispatch: the GPU reads the draw arguments out
of `buf` at `offset`, so a count produced on the device never has to reach the host.
"""
function draw_primitives_indirect!(rce::MTL.MTLRenderCommandEncoderLike, prim::MTL.MTLPrimitiveType,
                                   buf::MTL.MTLBuffer, offset::Integer = 0)
    @objc [rce::id{MTL.MTLRenderCommandEncoder} drawPrimitives:prim::MTL.MTLPrimitiveType
                                                indirectBuffer:buf::id{MTL.MTLBuffer}
                                                indirectBufferOffset:offset::NSUInteger]::Nothing
end

"""
    draw_mesh_threadgroups!(rce, groups, object, mesh)

A mesh pipeline's draw. There is no vertex count and no primitive type: the mesh
stage WRITES the primitives, and its `MeshConfig` already said what they are, so all
the host supplies is how many threadgroups to run and how wide each stage's
threadgroup is. `object` is the object stage's threadgroup size and is ignored by a
pipeline without one, but Metal still requires a valid size; one thread is the
smallest.
"""
function draw_mesh_threadgroups!(rce::MTL.MTLRenderCommandEncoderLike, groups::MTL.MTLSize,
                                 object::MTL.MTLSize, mesh::MTL.MTLSize)
    @objc [rce::id{MTL.MTLRenderCommandEncoder} drawMeshThreadgroups:groups::MTL.MTLSize
                                                threadsPerObjectThreadgroup:object::MTL.MTLSize
                                                threadsPerMeshThreadgroup:mesh::MTL.MTLSize]::Nothing
end
