# Rasterisation on Metal: the resources behind Mantle's `Framebuffer`,
# `Texture2D`, `Sampler` and `CompiledGraphicsPipeline`, pipeline state
# translation, and the render-pass verbs. The graph decides what is drawn and in
# what order; this records it.

const MTLm = Metal.MTL

# One method per state value, not a `Dict`, so an unhandled value is a
# `MethodError`, not a `KeyError` mid-frame.

mtl_cull(::Mantle.NoCull)    = MTLm.MTLCullModeNone
mtl_cull(::Mantle.CullBack)  = MTLm.MTLCullModeBack
mtl_cull(::Mantle.CullFront) = MTLm.MTLCullModeFront

mtl_primitive(::Mantle.TriangleList)  = MTLm.MTLPrimitiveTypeTriangle
mtl_primitive(::Mantle.TriangleStrip) = MTLm.MTLPrimitiveTypeTriangleStrip
mtl_primitive(::Mantle.LineList)      = MTLm.MTLPrimitiveTypeLine
mtl_primitive(::Mantle.PointList)     = MTLm.MTLPrimitiveTypePoint

mtl_depth_compare(::Mantle.DepthLess)    = MTLm.MTLCompareFunctionLess
mtl_depth_compare(::Mantle.DepthLessEq)  = MTLm.MTLCompareFunctionLessEqual
mtl_depth_compare(::Mantle.DepthLessEqReadOnly) = MTLm.MTLCompareFunctionLessEqual
mtl_depth_compare(::Mantle.DepthGreater) = MTLm.MTLCompareFunctionGreater
mtl_depth_compare(::Mantle.DepthAlways)  = MTLm.MTLCompareFunctionAlways
mtl_depth_compare(::Mantle.DepthOff)     = MTLm.MTLCompareFunctionAlways

depth_writes(::Mantle.DepthMode)  = true
depth_writes(::Mantle.DepthOff)   = false
depth_writes(::Mantle.DepthLessEqReadOnly) = false

"""Configure one colour attachment's blending from the portable `BlendMode`."""
function apply_blend!(att, ::Mantle.Opaque)
    att.blendingEnabled = false
    return att
end
function apply_blend!(att, ::Mantle.NoWrite)
    att.blendingEnabled = false
    att.writeMask = MTLm.MTLColorWriteMaskNone
    return att
end
function apply_blend!(att, ::Mantle.AlphaBlend)
    att.blendingEnabled = true
    att.sourceRGBBlendFactor        = MTLm.MTLBlendFactorSourceAlpha
    att.destinationRGBBlendFactor   = MTLm.MTLBlendFactorOneMinusSourceAlpha
    att.sourceAlphaBlendFactor      = MTLm.MTLBlendFactorOne
    att.destinationAlphaBlendFactor = MTLm.MTLBlendFactorOneMinusSourceAlpha
    return att
end
function apply_blend!(att, ::Mantle.Premultiplied)
    att.blendingEnabled = true
    att.sourceRGBBlendFactor        = MTLm.MTLBlendFactorOne
    att.destinationRGBBlendFactor   = MTLm.MTLBlendFactorOneMinusSourceAlpha
    att.sourceAlphaBlendFactor      = MTLm.MTLBlendFactorOne
    att.destinationAlphaBlendFactor = MTLm.MTLBlendFactorOneMinusSourceAlpha
    return att
end
function apply_blend!(att, ::Mantle.Additive)
    att.blendingEnabled = true
    att.sourceRGBBlendFactor        = MTLm.MTLBlendFactorOne
    att.destinationRGBBlendFactor   = MTLm.MTLBlendFactorOne
    att.sourceAlphaBlendFactor      = MTLm.MTLBlendFactorOne
    att.destinationAlphaBlendFactor = MTLm.MTLBlendFactorOne
    return att
end

function apply_attachments!(desc, color_formats, depth_format, blend)
    for (i, fmt) in enumerate(color_formats)
        att = desc.colorAttachments[i]
        att.pixelFormat = fmt
        apply_blend!(att, Mantle.attachmentblend(blend, i,
                     fmt in (MTLm.MTLPixelFormatR32Uint, MTLm.MTLPixelFormatRG32Uint)))
    end
    depth_format === nothing || (desc.depthAttachmentPixelFormat = depth_format)
    return desc
end

function depthstate(dev, depth, depth_format)
    (depth_format === nothing || depth isa Mantle.DepthOff) && return nothing
    dd = MTLm.MTLDepthStencilDescriptor()
    dd.depthCompareFunction = mtl_depth_compare(depth)
    dd.depthWriteEnabled = depth_writes(depth)
    return MTLm.MTLDepthStencilState(dev, dd)
end

"""A 2D texture on Metal. The element type is Mantle's spelling of the format."""
struct MetalTexture2D{T} <: Mantle.Texture2D{T}
    tex::MTLm.MTLTexture
    width::Int
    height::Int
end

Base.size(t::MetalTexture2D) = (t.width, t.height)
Mantle.mipmapped(t::MetalTexture2D) = t.tex.mipmapLevelCount > 1

"""A sampler on Metal."""
struct MetalSampler <: Mantle.Sampler
    state::MTLm.MTLSamplerState
end

"""
    mtlsampledformat(T) -> MTLPixelFormat

The pixel format of a sampled texture of `T`. Differs from `mtlformat` (render
targets) only for scalar floats: `Float32` is `R32Float` here, not `Depth32Float`,
which a sampler rejects.
"""
mtlsampledformat(::Type{Float32}) = MTLm.MTLPixelFormatR32Float
mtlsampledformat(::Type{Float16}) = MTLm.MTLPixelFormatR16Float
mtlsampledformat(@nospecialize(T::Type)) = mtlformat(T)

"""
    Texture2D(backend, data::AbstractMatrix; mipmap = false)

A sampled 2D texture holding `data` (host or device), with the element type
naming the format. `data[x, y]`: the first index is horizontal, as in GLMakie and
the Vulkan backend. `Shared` storage, not a buffer-backed linear texture.
`mipmap = true` regenerates the full mip chain from level 0 at every upload.
"""
function Mantle.Texture2D(::Metal.MetalBackend, data::AbstractMatrix{T}; mipmap::Bool = false) where {T}
    dev = Metal.device()
    w, h = size(data)          # first index is x, second is y
    desc = MTLm.MTLTextureDescriptor(mtlsampledformat(T), w, h, mipmap)
    desc.usage = MTLm.MTLTextureUsageShaderRead
    desc.storageMode = MTLm.MTLStorageModeShared
    t = MetalTexture2D{T}(MTLm.MTLTexture(dev, desc), w, h)
    Mantle.upload_texture_data!(t, data)
    return t
end

"""
    upload_texture_data!(tex::MetalTexture2D, data)

New texels for an existing texture, from the host or the device, through a
blit on Mantle's queue rather than `replace_region!`, so a frame still sampling
the texture is not overwritten. The matrix's column-major bytes are uploaded as
they are: one Julia column is one image row. Waits for completion.
"""
function Mantle.upload_texture_data!(t::MetalTexture2D{T}, data::AbstractMatrix{T}) where {T}
    size(data) == size(t) || throw(DimensionMismatch(
        "upload_texture_data!: $(size(data)) texels for a $(size(t)) texture"))
    # `collect` for anything without a dense host pointer (views, adjoints).
    rows = data isa Matrix{T} ? data : collect(data)
    nbytes = length(rows) * sizeof(T)
    staging = GC.@preserve rows MTLm.MTLBuffer(Metal.device(), nbytes, pointer(rows);
                                               storage = Metal.SharedStorage)
    blit_into_texture!(t, staging, 0)
    return t
end

# Device source: blit from the array's own buffer. A separate method because an
# `MtlArray` is a `DenseMatrix` whose `pointer` is a device address.
function Mantle.upload_texture_data!(t::MetalTexture2D{T}, data::Metal.MtlMatrix{T}) where {T}
    size(data) == size(t) || throw(DimensionMismatch(
        "upload_texture_data!: $(size(data)) texels for a $(size(t)) texture"))
    # The kernel that wrote `data` may be on this task's own Metal.jl queue (no
    # `adoptqueue!`), which `framebuffer!` does not order against.
    Metal.synchronize()
    p = pointer(data)
    blit_into_texture!(t, p.buffer, p.offset)
    return t
end

"""Copy a whole texture's worth of texels from `src` at byte `soff`, on the queue."""
function blit_into_texture!(t::MetalTexture2D{T}, src::MTLm.MTLBuffer, soff::Integer) where {T}
    w, h = size(t)
    # Behind the device's uncommitted batch: commit order is execution order.
    cb = framebuffer!(Device(MetalAPI()))
    MTLm.MTLBlitCommandEncoder(cb) do enc
        MTLm.append_copy!(enc, t.tex, MTLm.MTLOrigin(0, 0, 0), MTLm.MTLSize(w, h, 1),
                          src, soff, w * sizeof(T), 0)
        t.tex.mipmapLevelCount > 1 && MTLm.append_generate_mipmaps!(enc, t.tex)
    end
    # Waited for, so the caller may reuse its array on return.
    submitwait!(cb)
    cb.status == MTLm.MTLCommandBufferStatusCompleted ||
        error("Metal texture upload failed: $(cb.status)")
    return nothing
end

"""
    Sampler(backend; filter = :linear, wrap = :repeat)

How a texture is sampled; the same keywords as the Vulkan backend.
`filter` is `:nearest` or `:linear`, `wrap` is `:repeat`, `:clamp` or `:mirror`.
"""
function Mantle.Sampler(::Metal.MetalBackend; filter::Symbol = :linear,
                        wrap::Symbol = :repeat)
    f = filter === :nearest ? MTLm.MTLSamplerMinMagFilterNearest :
        filter === :linear  ? MTLm.MTLSamplerMinMagFilterLinear :
        error("unknown filter :$filter; expected :nearest or :linear")
    a = wrap === :repeat        ? MTLm.MTLSamplerAddressModeRepeat :
        wrap === :clamp        ? MTLm.MTLSamplerAddressModeClampToEdge :
        wrap === :mirror       ? MTLm.MTLSamplerAddressModeMirrorRepeat :
        error("unknown wrap :$wrap; expected :repeat, :clamp or :mirror")
    desc = MTLm.MTLSamplerDescriptor()
    desc.minFilter = f
    desc.magFilter = f
    desc.mipFilter = MTLm.MTLSamplerMipFilterLinear
    desc.sAddressMode = a
    desc.tAddressMode = a
    desc.normalizedCoordinates = true
    return MetalSampler(MTLm.MTLSamplerState(Metal.device(), desc))
end

"""
An offscreen render target: a `Shared` colour texture and an optional depth one.
Not buffer-backed: rendering into a linear texture on an Apple GPU silently writes
nothing, so readback uses `getBytes!`.
"""
struct MetalFramebuffer <: Mantle.Framebuffer
    color::MTLm.MTLTexture
    depth::Union{Nothing,MTLm.MTLTexture}
    width::Int
    height::Int
    color_format::MTLm.MTLPixelFormat
    depth_format::MTLm.MTLPixelFormat
end

"""
    Framebuffer(backend, width, height; depth = true, color_format = nothing)

Allocate an offscreen target. `color_format` is a Julia element type lowered by
`mtlformat` (default `BGRA8Unorm`); depth is `Depth32Float`.
"""
function Mantle.Framebuffer(be::Metal.MetalBackend, width::Integer, height::Integer;
                            depth::Bool = true, color_format = nothing)
    dev = Device(be).dev
    cfmt = color_format === nothing ? MTLm.MTLPixelFormatBGRA8Unorm :
                                      mtlformat(color_format)
    cdesc = MTLm.MTLTextureDescriptor(cfmt, width, height, false)
    cdesc.usage = MTLm.MTLTextureUsageRenderTarget | MTLm.MTLTextureUsageShaderRead
    cdesc.storageMode = MTLm.MTLStorageModeShared
    ctex = MTLm.MTLTexture(dev, cdesc)

    dfmt = MTLm.MTLPixelFormatDepth32Float
    dtex = nothing
    if depth
        ddesc = MTLm.MTLTextureDescriptor(dfmt, width, height, false)
        ddesc.usage = MTLm.MTLTextureUsageRenderTarget
        # Private: depth is never read back.
        ddesc.storageMode = MTLm.MTLStorageModePrivate
        dtex = MTLm.MTLTexture(dev, ddesc)
    end
    return MetalFramebuffer(ctex, dtex, Int(width), Int(height), cfmt, dfmt)
end

"""
    mtlformat(T) -> MTLPixelFormat

Metal's render-target format for a Julia element type: the table in
`runtime/format.jl`, as `vkformat` on Vulkan. `Float32` is depth. Matched on the
type, not `nameof(T)`, so foreign types named `RGBA` do not match.
"""
function mtlformat(@nospecialize(T::Type); srgb::Bool = false)
    # Depth, as `vkformat`'s `D32_SFLOAT`; `R32Float` would be a colour target.
    T === Float32 && return MTLm.MTLPixelFormatDepth32Float
    T === UInt32 && return MTLm.MTLPixelFormatR32Uint
    T <: StaticVector{2,UInt32} && return MTLm.MTLPixelFormatRG32Uint
    T <: Real && error("no Metal pixel format for the scalar $T")
    T === BGRA{N0f8}    && return srgb ? MTLm.MTLPixelFormatBGRA8Unorm_sRGB :
                                         MTLm.MTLPixelFormatBGRA8Unorm
    T === RGBA{N0f8}    && return srgb ? MTLm.MTLPixelFormatRGBA8Unorm_sRGB :
                                         MTLm.MTLPixelFormatRGBA8Unorm
    # Float formats have no sRGB variant; they hold linear values.
    T === RGBA{Float16} && return MTLm.MTLPixelFormatRGBA16Float
    T === RGBA{Float32} && return MTLm.MTLPixelFormatRGBA32Float

    # Tuple spellings, as portable shaders write them.
    T === NTuple{4,Float32} && return MTLm.MTLPixelFormatRGBA32Float
    T === NTuple{2,Float32} && return MTLm.MTLPixelFormatRG32Float
    T === NTuple{4,Float16} && return MTLm.MTLPixelFormatRGBA16Float
    T === NTuple{2,Float16} && return MTLm.MTLPixelFormatRG16Float
    T === NTuple{4,UInt8}   && return srgb ? MTLm.MTLPixelFormatRGBA8Unorm_sRGB :
                                             MTLm.MTLPixelFormatRGBA8Unorm
    T === UInt8             && return MTLm.MTLPixelFormatR8Unorm
    # No three-channel formats exist on Metal; callers pad to four.
    error("no Metal pixel format for $T — extend `mtlformat` in metal/graphics.jl")
end

"""A compiled Metal render pipeline, plus the state a draw still has to set."""
struct MetalCompiledGraphicsPipeline <: Mantle.CompiledGraphicsPipeline
    state::MTLm.MTLRenderPipelineState
    depth_state::Union{Nothing,MTLm.MTLDepthStencilState}
    primitive::MTLm.MTLPrimitiveType
    cull::MTLm.MTLCullMode
    # False for depth-only pipelines; Metal rejects binding to a missing stage.
    has_fragment::Bool
    # An `MTLFunction` does not keep its library alive; releasing the library
    # breaks the pipeline's shaders.
    libs::Vector{Any}
end

"""
    stage_output_type(pipeline, stage) -> Type

The struct a stage writes. For the vertex stage this is `Mantle.outputtype`, a
`NamedTuple` type with the clip position prepended, which Metal.jl's
`stage_outputs` reads directly.
"""
stage_output_type(p::Mantle.GraphicsPipeline, ::Val{:vertex}, ncolor::Int = 1) =
    Mantle.outputtype(p.vertex)

"""
For the fragment stage: one field per colour attachment of the pass (each field
is an `air.render_target`), so the count comes from the pass, not the shader.
"""
stage_output_type(::Mantle.GraphicsPipeline, ::Val{:fragment}, ncolor::Int = 1) =
    NamedTuple{ntuple(i -> Symbol(:color, i), ncolor),
               NTuple{ncolor, NTuple{4,Float32}}}

# Integer attachments carry exact IDs rather than float colour values.
fragment_attachment_type(fmt) = fmt == MTLm.MTLPixelFormatR32Uint ? UInt32 :
    fmt == MTLm.MTLPixelFormatRG32Uint ? NTuple{2,UInt32} : NTuple{4,Float32}
fragment_output_type(formats) = NamedTuple{
    ntuple(i -> Symbol(:color, i), length(formats)),
    Tuple{map(fragment_attachment_type, formats)...}}

# Portable shaders return their outputs: a vertex stage returns
# `(position = …, varyings…)`, a fragment stage takes the varyings first and
# returns one value per attachment. AIR stages write through a trailing output
# pointer and read each varying as a tagged parameter; the `@generated` wrappers
# below translate at compile time, as Lava's `VertexWrapper`/`FragmentWrapper` do.

"""
    to_stage_field(T, v)

Convert `v` to the declared field type `T`. A one-field wrapper of `T` (`Vec2f`
for `NTuple{2,Float32}`) is unwrapped with `getfield`, since `convert` has no
method between them and would throw from device code.
"""
@generated function to_stage_field(::Type{T}, v) where {T}
    v === T && return :v
    if isconcretetype(v) && !isprimitivetype(v) && fieldcount(v) == 1 &&
       fieldtype(v, 1) === T
        return :(Base.getfield(v, 1))
    end
    return :(convert(T, v))
end

"""
    stage_fragment_out(Out, r)

A fragment stage's declared outputs from its return value: a tuple of one value
per attachment, or a single value. A colour is itself a tuple, so the two are
told apart by element type: per-attachment tuples do not hold `Real`s.
"""
@generated function stage_fragment_out(::Type{Out}, r) where {Out}
    per_target = r <: Tuple && fieldcount(r) == fieldcount(Out) &&
                 fieldcount(r) > 0 && !(fieldtype(r, 1) <: Real)
    n = fieldcount(Out)
    vals = per_target ?
        Expr(:tuple, (:(to_stage_field($(fieldtype(Out, i)), r[$i])) for i in 1:n)...) :
        Expr(:tuple, :(to_stage_field($(fieldtype(Out, 1)), r)))
    return :(Out($vals))
end

"""
All of a stage's baked arguments, bound as one `setBytes` block and splatted back
into the body's arguments by the stage wrappers. One buffer slot per argument
would cap a stage at Metal's 31 buffer slots.
"""
struct PackedArgs{T<:Tuple}
    args::T
end

# `setBytes` takes at most 4 KB; past that the block would need a buffer of its own.
function packargs(d::Tuple)
    isempty(d) && return ()
    p = PackedArgs(d)
    sizeof(p) <= 4096 || throw(ArgumentError(
        "a stage's arguments are $(sizeof(p)) bytes, and Metal binds at most 4096 " *
        "bytes inline; pass the larger values in a buffer"))
    return (p,)
end

# Defined before the `@generated` wrappers that call it: a generator runs in the
# world of its definition.
"""The body's arguments, expressions for a `@generated` wrapper: a `PackedArgs` splatted in place."""
function unpacked(argtypes, idxs)
    exprs = Any[]
    for i in idxs
        T = argtypes[i]
        if T <: PackedArgs
            append!(exprs, (:(args[$i].args[$j]) for j in 1:fieldcount(fieldtype(T, 1))))
        else
            push!(exprs, :(args[$i]))
        end
    end
    return exprs
end

"""The types the body is called with: `bufs` with a `PackedArgs` splatted in place, as `unpacked` does to the values."""
bodytypes(bufs::Type{<:Tuple}) =
    Tuple{(U for T in fieldtypes(bufs) for U in (T <: PackedArgs ? fieldtypes(fieldtype(T, 1)) : (T,)))...}

"""
A vertex shader that returns `(position = …, varyings…)`. `VID` is whether the
body takes a leading `KernelInterface.VertexIndex`, decided on the host
(`KernelInterface.wantsvertexindex`) because a `@generated` function may not call
`hasmethod`.
"""
struct MetalVertexStage{F, Out, VID} end

# Two-parameter form, for a body that calls `vertex_index()`.
MetalVertexStage{F,Out}() where {F,Out} = MetalVertexStage{F,Out,false}()

@generated function (::MetalVertexStage{F,Out,VID})(args::Vararg{Any,N}) where {F,Out,VID,N}
    bufs = unpacked(args, 1:(N - 1))
    call = VID ? Expr(:call, :(F.instance), :(KI.VertexIndex(KI.vertex_index())), bufs...) :
                 Expr(:call, :(F.instance), bufs...)
    # Fields by name, in declaration order; `position` is mirrored into Metal's
    # clip space.
    vals = Expr(:tuple, (n === :position ?
                         :(flip_clip(to_stage_field($(fieldtype(Out, n)),
                                                    Base.getfield(r, :position)))) :
                         :(to_stage_field($(fieldtype(Out, n)),
                                          Base.getfield(r, $(QuoteNode(n)))))
                         for n in fieldnames(Out))...)
    quote
        Base.@_inline_meta
        r = $call
        Base.unsafe_store!(args[$N]::StageOut{Out}, Out($vals))
        return nothing
    end
end

"""
A fragment shader that takes the varyings as a NamedTuple and returns its
colours. Parameters are buffers, varying markers, textures and samplers, then the
output pointer: buffers stay first so their `air.buffer` indices are the slots
`record_draw!` binds, and `stage_return!` takes the last parameter as the output.
"""
struct MetalFragmentStage{F, VIn, Out, NT} end

# Three-parameter form, for a stage that samples nothing.
MetalFragmentStage{F,VIn,Out}() where {F,VIn,Out} = MetalFragmentStage{F,VIn,Out,0}()

@generated function (::MetalFragmentStage{F,VIn,Out,NT})(args::Vararg{Any,N}) where {F,VIn,Out,NT,N}
    nv = fieldcount(VIn)
    # Each texture adds a texture and a sampler parameter after the varyings
    # (`texture_markers`). The body takes neither: `sample_texture_2d` reaches
    # them through the lowering in `Metal/src/compiler/texture.jl`.
    ntex = 2 * NT
    nbuf = N - 1 - nv - ntex
    nbuf >= 0 || error("a fragment stage with $nv varyings and $NT textures needs " *
                       "at least $(nv + ntex + 1) parameters, got $N")
    ins = Expr(:tuple, (:(args[$(nbuf + i)].value) for i in 1:nv)...)
    call = Expr(:call, :(F.instance), :(VIn($ins)), unpacked(args, 1:nbuf)...)
    quote
        Base.@_inline_meta
        r = $call
        Base.unsafe_store!(args[$N]::StageOut{Out}, stage_fragment_out(Out, r))
        return nothing
    end
end

"""
One `Varying` marker per declared varying, in declaration order. Stages link
varyings by mangled name; the stable order keeps argument metadata reproducible.
"""
varying_markers(VIn::Type) =
    Tuple(Metal.Varying{n, fieldtype(VIn, n)} for n in fieldnames(VIn))

"""
    texture_markers(n) -> Tuple

Entry parameters for `n` sampled textures: `n` texture pointers, then `n`
sampler pointers, in binding order (AIR textures are entry parameters). They must
follow the varyings: textures before the interpolated inputs crash the Metal
compiler service in `AGCLLVMUserFragmentShader::setupShaderInputs`
(`XPC_ERROR_CONNECTION_INTERRUPTED` at pipeline creation).
"""
texture_markers(n::Integer) =
    (ntuple(_ -> Metal.Texture2DPtr{Float32}, n)..., ntuple(_ -> Metal.SamplerPtr, n)...)

"""
The fragment stage's varyings as a `NamedTuple` type (`Mantle.fragmentinputtype`),
for graphics and mesh pipelines. May be empty, e.g. for a shadow pass.
"""
varying_type(p::Union{Mantle.GraphicsPipeline,Mantle.MeshPipeline}) = Mantle.fragmentinputtype(p)

"""
    stage_signatures(p, ncolor, vert_bufs, frag_bufs) -> (vfn, ffn, vert_tt, frag_tt)

The two wrapped stage callables to compile and the Julia signature of each.
`ffn` and `frag_tt` are `nothing` when there is no colour attachment.
"""
function stage_signatures(p::Mantle.GraphicsPipeline, ncolor::Int,
                          vert_bufs::Type, frag_bufs::Type; color_formats = nothing)
    VIn  = varying_type(p)
    VOut = stage_output_type(p, Val(:vertex))
    vf = Mantle.stagefunction(p.vertex)
    # A body declaring a leading `VertexIndex` is handed one.
    vid = KI.wantsvertexindex(vf, fieldtypes(bodytypes(vert_bufs)))
    vfn = MetalVertexStage{typeof(vf), VOut, vid}()
    vert_tt = Tuple{vert_bufs.parameters..., StageOut{VOut}}
    # No colour attachment: a nil `fragmentFunction`; AIR has no empty output struct.
    ncolor == 0 && return vfn, nothing, vert_tt, nothing
    FOut = color_formats === nothing ? stage_output_type(p, Val(:fragment), ncolor) :
        fragment_output_type(color_formats)
    ntex = Mantle.ntextures(p.fragment)
    ffn = MetalFragmentStage{typeof(Mantle.stagefunction(p.fragment)), VIn, FOut, ntex}()
    frag_tt = Tuple{frag_bufs.parameters..., varying_markers(VIn)...,
                    texture_markers(ntex)..., StageOut{FOut}}
    return vfn, ffn, vert_tt, frag_tt
end

"""
    compile_pipeline(device, pipeline, color_formats, depth_format, vert_bufs, frag_bufs)

Compile a portable `GraphicsPipeline` into a Metal render pipeline state.

`vert_bufs`/`frag_bufs` are Tuple types of each stage's buffer arguments only;
the rest of the signature follows from the pipeline. Cached on the compiled
stages, the formats and the pipeline state.
"""
function compile_pipeline(d::MetalDevice, p::Mantle.GraphicsPipeline,
                          color_formats::Vector{MTLm.MTLPixelFormat},
                          depth_format::Union{Nothing,MTLm.MTLPixelFormat},
                          vert_bufs::Type, frag_bufs::Type)
    # Geometry pipelines normally go through `compile_draw`, which lowers them
    # onto a mesh pipeline (`metal/mesh.jl`); this compiles native stages only.
    be = Metal.MetalBackend()
    p.geometry === nothing || Mantle.supports_geometry_stage(be) ||
        error("`compile_pipeline` compiles each stage of the pipeline as itself, " *
              "and this backend has no geometry stage to compile one into. Apple's " *
              "replacement is the mesh pipeline, which this backend DOES run: go " *
              "through `Mantle.compile_draw`, which lowers a geometry pipeline " *
              "with `Mantle.lower_geometry_to_mesh` and compiles the mesh " *
              "pipeline that comes out. Ask `supports_geometry_stage(backend)` " *
              "only if you need to know which of the two ran.")
    (p.tess_control === nothing && p.tess_eval === nothing) ||
        Mantle.supports_tessellation(be) ||
        error("this backend has no tessellation; ask " *
              "`supports_tessellation(backend)` before building a pipeline " *
              "with it.")

    dev = d.dev
    vfn, ffn, vert_tt, frag_tt =
        stage_signatures(p, length(color_formats), vert_bufs, frag_bufs; color_formats)
    vstage = compile_stage_function(dev, vfn, vert_tt, :vertex,
                                    stagename(Mantle.stagefunction(p.vertex), "_vs"))
    fstage = ffn === nothing ? nothing :
        compile_stage_function(dev, ffn, frag_tt, :fragment,
                               stagename(Mantle.stagefunction(p.fragment), "_fs"))

    # Native stages carry compiled code identity, independent of captured data.
    key = (:raster, CacheIdentity((vstage, fstage)), Tuple(color_formats),
           depth_format, pipeline_state_key(p))
    return get!(d.pipelines, key) do
        vfun, vlib = vstage
        ffun, flib = fstage === nothing ? (nothing, nothing) : fstage

        desc = MTLm.MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfun
        ffun === nothing || (desc.fragmentFunction = ffun)
        # Attachments in pass order, matching `air.render_target <i>`.
        Mantle.checkdepthtarget(p.depth, depth_format !== nothing)
        apply_attachments!(desc, color_formats, depth_format, p.blend)
        state = MTLm.MTLRenderPipelineState(dev, desc)
        dstate = depthstate(dev, p.depth, depth_format)

        compiled = MetalCompiledGraphicsPipeline(state, dstate, mtl_primitive(p.topology),
                                                 mtl_cull(p.cull), ffun !== nothing,
                                                 Any[l for l in (vlib, flib) if l !== nothing])
        return compiled
    end::MetalCompiledGraphicsPipeline
end

"""
Stage functions compiled from one `CodeInstance` of a Julia function, by
`(device, stage, name)`. Held in GPUCompiler's `cached_results`, so editing the
function or anything inlined into it compiles the stage again.
"""
mutable struct MetalStages
    stages::Dict{Any,Tuple{Any,Any}}
    MetalStages() = new(Dict{Any,Tuple{Any,Any}}())
end

"""
`name` with every non-identifier character replaced by `_`, which is how the
entry point is named in the metallib (`#332` becomes `_332`).
"""
identifier(name::AbstractString) =
    map(c -> (isletter(c) || isdigit(c) || c == '_') ? c : '_', String(name))

"""The entry name of shader `f`'s `suffix` stage."""
stagename(f, suffix::AbstractString) = identifier(string(nameof(f))) * suffix

"""
    compile_stage_function(device, f, tt, stage, name) -> (MTLFunction, MTLLibrary)

Compile one Julia shader into a Metal stage function, cached per `CodeInstance`.
The caller must keep the library: an `MTLFunction` does not keep it alive, and a
released library makes the pipeline draw nothing.
"""
function compile_stage_function(dev::MTLm.MTLDevice, f, tt::Type, stage::Symbol, name::String)
    # `always_inline`: textures and samplers are entry parameters
    # (`lower_texture_bindings!`), so every sampling call must be inlined into the
    # entry; the optimiser alone does not guarantee it.
    cfg = Metal.compiler_config(dev; stage, name, always_inline = true)
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt), cfg)
    key = (dev, stage, name)
    stages = Metal.GPUCompiler.cached_results(MetalStages, job)
    if stages !== nothing
        compiled = get(stages.stages, key, nothing)
        compiled === nothing || return compiled
    end
    lib = MTLm.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib)
    compiled = (MTLm.MTLFunction(lib, name), lib)
    # Compiling the job made its `CodeInstance`, so this finds one.
    stages === nothing && (stages = Metal.GPUCompiler.cached_results(MetalStages, job))
    stages.stages[key] = compiled
    return compiled
end

# `draw!` is core's (`graphics/record.jl`), over `compile_draw` and `record_draw!`.

# Submission: render passes, copies, presents and readbacks share the compute
# queue and each opens one command buffer, committed before the call returns.
# Command buffers on one queue run in commit order, which is the order the
# graph's barrier phase assumes; two queues have no mutual order (an indirect
# draw then read an unwritten count and hung the GPU). Metal.jl's open compute
# batch is flushed first so ours runs behind it.

"""
    framebuffer!(dev) -> MTLCommandBuffer

A fresh command buffer on the device's queue, behind the compute Metal.jl has
batched and not committed. Given a raw `MTLDevice`, uses the cached
`Device(MetalAPI())`.
"""
framebuffer!(::MTLm.MTLDevice) = framebuffer!(Device(MetalAPI()))

function framebuffer!(d::MetalDevice)
    bq = batchqueue(d)
    Metal.has_open_batch(bq) && Metal.flush!(bq)
    return MTLm.MTLCommandBuffer(bq.queue)
end

"""Commit a command buffer from `framebuffer!`; every caller ends with this or
`submitwait!`. `Metal.submit_command_buffer` commits Metal.jl's open batch first
and claims the bound arrays (`metal_buffer`) for this queue, so host reads of
them wait for this buffer."""
function commit!(cb::MTLm.MTLCommandBuffer)
    Metal.submit_command_buffer(() -> MTLm.commit!(cb), cb, true)
    committed!(cb)
    return cb
end

"""Commit and wait; required before reading bytes back."""
function submitwait!(cb::MTLm.MTLCommandBuffer)
    commit!(cb)
    MTLm.wait_completed(cb)
    return nothing
end

# Per-pass GPU time from each command buffer's `GPUStartTime`/`GPUEndTime`
# (seconds). Committed buffers are collected while profiling and drained by
# `gpupasstime!`, which submits and waits at every pass boundary.
const COMMITTED = MTLm.MTLCommandBuffer[]

# Set by `makeprofiler`. Held command buffers are not recycled by Metal, so
# collecting without a drain makes frame time grow with frame count.
const COLLECT = Ref(false)

# Collection is process-wide but draining is per profiled plan, so an unprofiled
# plan's buffers are capped here. A frame commits fewer than ten.
const COMMITTED_MAX = 64

"""Remember a command buffer so a profiled run can ask what it cost."""
function committed!(cb)
    COLLECT[] || return cb
    length(COMMITTED) >= COMMITTED_MAX && popfirst!(COMMITTED)
    push!(COMMITTED, cb)
    return cb
end

"""
Enable per-pass GPU collection with the host profiler when `profile`. Collection
is process-wide, so another plan's buffers on the same queue are counted into the
profiled plan's next pass.
"""
function Mantle.makeprofiler(dev::MetalDevice, passes, profile::Bool)
    profile || return nothing
    COLLECT[] = true
    return Mantle.hostprofiler(passes)
end

"""
Nanoseconds of GPU time for everything committed since the last call; `0.0` if
nothing was. Flushes and waits. Buffers that never ran contribute nothing.
Recorded passes report no GPU time (`passend!`).
"""
function Mantle.gpupasstime!(d::MetalDevice)
    bq = batchqueue(d)
    # Metal.jl's open batch counts too. Retained, because the batch releases its
    # borrowed reference when the flush commits it.
    open_cb = bq.backend.cmdbuf
    open_cb === nothing || push!(COMMITTED, MTLm.retained(open_cb))
    Metal.flush!(bq)
    isempty(COMMITTED) && return 0.0
    total = 0.0
    for cb in COMMITTED
        MTLm.wait_completed(cb)
        s, e = cb.GPUStartTime, cb.GPUEndTime
        (s > 0 && e > s) && (total += e - s)
    end
    empty!(COMMITTED)
    return total * 1e9          # `GPUStartTime` is in seconds
end

"""
    readback_framebuffer(fb) -> Matrix{<:Colorant}

The colour attachment as a `width` x `height` matrix of
`eltypeof(fb.color_format)`, the same colorant and shape as on Vulkan; the
colorant carries the channel order. Waits for outstanding GPU work, then reads
with `getBytes!`.
"""
function Mantle.readback_framebuffer(fb::MetalFramebuffer)
    # `getBytes!` is a host copy that does not wait for the queue.
    Metal.synchronize()
    # The colorant's layout matches the texel's, so bytes are read directly.
    px = Matrix{eltypeof(fb.color_format)}(undef, fb.width, fb.height)
    GC.@preserve px MTLm.getBytes!(pointer(px), fb.color, fb.width * sizeof(eltype(px)),
                                   MTLm.MTLRegion(MTLm.MTLOrigin(0, 0, 0),
                                                  MTLm.MTLSize(fb.width, fb.height, 1)))
    return px
end

supports_graphics(::Metal.MetalBackend) = true
supports_graphics(::MetalDevice) = true

# The render-pass verbs declared in `graphics/commands.jl`, driven by core's
# `runrenderpass!`.

"""What `begin_render_pass!` returns and `record_draw!`/`end_render_pass!` take."""
struct MetalPassHandle
    cmdbuf::MTLm.MTLCommandBuffer
    encoder::MTLm.MTLRenderCommandEncoder
end

# Draw arguments use the kernel-launch ABI: `mtlconvert` turns an `MtlArray`
# into an `MtlDeviceArray` (address plus dimensions) whose bytes are bound, so
# shaders can use `length` and bounds checks as on Vulkan.

"""
The device-side form of one stage's arguments, plus the buffers behind them,
which `record_draw!` passes to `use!`: a non-resident address reads as zeros.
A value that changes between frames must be a `GPURef` (`refuserefs`).
"""
mutable struct StageArgs
    source::Tuple      # the arguments last baked from
    device::Tuple      # what gets bound, byte for byte
    buffers::Vector{MTLm.MTLBuffer}
    # Reused staging for `setVertexBytes`, which takes a pointer; avoids a `Ref`
    # per argument per draw.
    scratch::Vector{UInt8}
end

bakearg(a) = Metal.mtlconvert(a)

"""What a baked argument is once it reaches the shader."""
argdevtype(a) = typeof(a)

function StageArgs(args)
    t = Tuple(args)
    bufs = MTLm.MTLBuffer[]
    for a in t
        b = metal_buffer(a)
        b === nothing || push!(bufs, b)
    end
    d = packargs(map(bakearg, t))
    return StageArgs(t, d, bufs, Vector{UInt8}(undef, scratchsize(d)))
end

# The widest argument; `rebake!` only ever grows the scratch.
scratchsize(d::Tuple) = isempty(d) ? 0 : maximum(sizeof(typeof(x)) for x in d)

"""
Re-bake `a` from this frame's `args` (a [`Mantle.DrawBinding`](@ref) can change
them). Unconditional: the same Julia object can have a new device address, since
`resize!` on an `MtlVector` reallocates its `MTLBuffer`.
"""
rebake!(a::StageArgs, args) = rebake!(a, Tuple(args))

# Everything is computed from the concrete `t` before storing: reading back the
# `::Tuple` fields mid-bake would make the work dynamic and allocate.
function rebake!(a::StageArgs, t::T) where {T<:Tuple}
    d = packargs(map(bakearg, t))
    n = scratchsize(d)
    length(a.scratch) < n && resize!(a.scratch, n)
    empty!(a.buffers)
    collectbuffers!(a.buffers, t)
    a.source = t
    a.device = d
    return a
end

"""
Push the device buffer of every argument that has one onto `bufs`. Unrolled,
like `bindall!`: iterating a heterogeneous tuple boxes each element.
"""
@generated function collectbuffers!(bufs, t::T) where {T<:Tuple}
    Expr(:block, Expr(:meta, :inline),
         (:(let b = metal_buffer(t[$i]); b === nothing || push!(bufs, b) end)
          for i in 1:fieldcount(T))...,
         :(return nothing))
end

"""One draw's compiled pipeline and its baked arguments."""
struct MetalCompiledDraw
    pipeline::MetalCompiledGraphicsPipeline
    vert::StageArgs
    frag::StageArgs
end

"""The Tuple type of one stage's device-side arguments."""
buffer_types(a::StageArgs) = Tuple{map(argdevtype, a.device)...}

# `bindings` is ignored: textures bind to encoder slots, so they do not affect
# pipeline compilation (Vulkan needs the layout at compile).
function Mantle.compile_draw(d::MetalDevice, p::Mantle.GraphicsPipeline,
                             color_formats, depth_format, vert_args, frag_args;
                             bindings = nothing)
    cfmts = MTLm.MTLPixelFormat[mtlformat(T) for T in color_formats]
    vert = StageArgs(vert_args)
    frag = StageArgs(frag_args)
    # Through `mtlformat`, so the pipeline matches the attachment's format.
    dfmt = depth_format === nothing ? nothing : mtlformat(depth_format)
    pipeline = compile_pipeline(d, p, cfmts, dfmt, buffer_types(vert), buffer_types(frag))
    return MetalCompiledDraw(pipeline, vert, frag)
end

function Mantle.begin_render_pass!(d::MetalDevice, targets, loads, depth, depth_load)
    rp = MTLm.MTLRenderPassDescriptor()
    for (i, t) in enumerate(targets)
        att = rp.colorAttachments[i]
        att.texture     = metal_texture(t)
        att.loadAction  = mtl_loadaction(loads[i])
        att.storeAction = MTLm.MTLStoreActionStore
        cv = Mantle.clearvalue(loads[i])
        cv === nothing || (att.clearColor = MTLm.MTLClearColor(cv[1], cv[2], cv[3], cv[4]))
    end
    if depth !== nothing
        da = rp.depthAttachment
        da.texture     = metal_texture(depth)
        da.loadAction  = mtl_loadaction(depth_load)
        da.storeAction = MTLm.MTLStoreActionStore
        dv = depth_load === nothing ? nothing : Mantle.depthclear(depth_load)
        dv === nothing || (da.clearDepth = Float64(dv))
    end
    # This pass's own command buffer; `end_render_pass!` commits it.
    cb  = framebuffer!(d.dev)
    enc = MTLm.MTLRenderCommandEncoder(cb, rp)
    return MetalPassHandle(cb, enc)
end

"""
Bind one stage's baked arguments with `setBytes:` (a stage takes the
`MtlDeviceArray` struct, not the data), after making every buffer behind them
resident.
"""
function bind_stage!(enc, a::StageArgs, setbytes!, stage::MTLm.MTLRenderStages)
    for b in a.buffers
        MTLm.use!(enc, b, MTLm.ReadUsage, stage)
    end
    bindall!(setbytes!, enc, a.device, a.scratch)
    return nothing
end

"""
Bind every argument of a stage. Unrolled by `@generated`: iterating the
`::Tuple` field would box each element; here one dynamic dispatch lands on the
concrete tuple type and the rest is static.
"""
@generated function bindall!(setbytes!, enc, d::D, scratch) where {D<:Tuple}
    Expr(:block, Expr(:meta, :inline),
         (:(bind_arg!(setbytes!, enc, d[$i], $i, scratch)) for i in 1:fieldcount(D))...,
         :(return nothing))
end

# Staged in the stage's `scratch`: `setVertexBytes` copies the bytes before it
# returns, so one reused area suffices.
function bind_arg!(setbytes!, enc, arg::T, i::Int, scratch::Vector{UInt8}) where {T}
    GC.@preserve scratch begin
        ptr = Base.unsafe_convert(Ptr{T}, pointer(scratch))
        unsafe_store!(ptr, arg)
        setbytes!(enc, reinterpret(Ptr{Cvoid}, ptr), sizeof(T), i)
    end
    return nothing
end

function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledDraw, args, count;
                             instances::Integer = 1, indices = nothing)
    # `args` is this frame's resolved list, shared by both stages; a stage with
    # arguments re-bakes from it every draw (`rebake!`).
    isempty(d.vert.device) || rebake!(d.vert, args)
    isempty(d.frag.device) || rebake!(d.frag, args)
    c = d.pipeline
    MTLm.set_pipeline!(h.encoder, c.state)
    c.depth_state === nothing ||
        MTLm.set_depth_stencil_state!(h.encoder, c.depth_state)
    MTLm.set_cull_mode!(h.encoder, c.cull)
    # Counter-clockwise: `flip_clip` mirrors y, reversing every triangle's
    # winding; otherwise back-face culling drops the front faces.
    MTLm.set_front_facing_winding!(h.encoder, MTLm.MTLWindingCounterClockwise)
    bind_stage!(h.encoder, d.vert, MTLm.set_vertex_bytes!, MTLm.MTLRenderStageVertex)
    # Metal rejects fragment bindings on a depth-only pipeline.
    c.has_fragment &&
        bind_stage!(h.encoder, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)
    draw_with_count!(h.encoder, c, count, instances, indices)
    return nothing
end

"""
    setviewport!(handle, x, y, width, height)

The viewport for the following draws, and a scissor over the same rectangle.
A negative `height` (Vulkan's flipped viewport, `y` naming the bottom edge) is
passed to `MTLViewport` unchanged, which interprets it the same way and undoes
the vertex stage's `flip_clip`. Only the scissor is normalised.
"""
function Mantle.setviewport!(h::MetalPassHandle, x::Real, y::Real, w::Real, hgt::Real)
    top = min(y, y + hgt)
    MTLm.set_viewport!(h.encoder,
        MTLm.MTLViewport(Float64(x), Float64(y), Float64(w), Float64(hgt), 0.0, 1.0))
    MTLm.set_scissor!(h.encoder,
        MTLm.MTLScissorRect(UInt(max(0, floor(Int, x))), UInt(max(0, floor(Int, top))),
                            UInt(max(1, ceil(Int, w))), UInt(max(1, ceil(Int, abs(hgt))))))
    return nothing
end

"""
    indexbuffer(::MetalDevice, indices) -> MtlVector{UInt32}

An index buffer as a device array, as on Vulkan (core's default returns a
`Mantle.Buffer`). Uses the pool's buffer storage mode: `MtlVector`'s default
private storage stages every host write through a blit, and index buffers are
rewritten from the host on every relayout.
"""
Mantle.indexbuffer(d::MetalDevice, indices::AbstractVector{UInt32}) =
    buffercopy(d, indices)

"""
    devicearray(backend, data) -> MtlVector

An array of `data` on Metal, in the pool's buffer storage mode rather than the
private default of `KernelAbstractions.allocate` (see `indexbuffer`).
"""
Mantle.devicearray(be::Metal.MetalBackend, data::AbstractArray) =
    buffercopy(Mantle.todevice(be), data)

# The storage mode is a type parameter of `MtlArray`, not a keyword.
function buffercopy(d::MetalDevice, data::AbstractArray{T,N}) where {T,N}
    S = constraintof(d, Buffers(), T)
    a = Metal.MtlArray{T,N,S}(undef, size(data))
    copyto!(a, data)
    return a
end

# A `MTLTexture` is both the view and the image.
Mantle.target_view(fb::MetalFramebuffer) = fb.color
Mantle.target_image(fb::MetalFramebuffer) = fb.color

Mantle.colorimage(fb::MetalFramebuffer) = fb.color
Mantle.depthimage(fb::MetalFramebuffer) = fb.depth

Mantle.target_extent(fb::MetalFramebuffer) = (fb.width, fb.height)
Mantle.target_format(fb::MetalFramebuffer) = fb.color_format
# The inverse of `mtlformat` over a framebuffer's colour formats, for core's
# `blittarget` and for `readback_framebuffer`.

eltypeof(f::MTLm.MTLPixelFormat) =
    f === MTLm.MTLPixelFormatBGRA8Unorm      ? BGRA{N0f8} :
    f === MTLm.MTLPixelFormatBGRA8Unorm_sRGB ? BGRA{N0f8} :
    f === MTLm.MTLPixelFormatRGBA8Unorm      ? RGBA{N0f8} :
    f === MTLm.MTLPixelFormatRGBA8Unorm_sRGB ? RGBA{N0f8} :
    f === MTLm.MTLPixelFormatRGBA16Float     ? RGBA{Float16} :
    f === MTLm.MTLPixelFormatRGBA32Float     ? RGBA{Float32} :
    error("no Julia element type recorded for $f; add it beside `mtlformat`")

"""
End the encoder and commit the pass's command buffer.
"""
function Mantle.end_render_pass!(h::MetalPassHandle)
    MTLm.endEncoding!(h.encoder)
    commit!(h.cmdbuf)
    return nothing
end

"""
    draw_with_count!(enc, c, n, instances = 1, indices = nothing)

A plain count draws directly; a `Commands` buffer draws indirectly. `instances`
and `indices` come only from hand-recorded passes. An index buffer is made
resident explicitly: no shader names it.
"""
function draw_with_count!(enc, c::MetalCompiledGraphicsPipeline, n::Integer,
                          instances::Integer = 1, indices = nothing)
    if indices === nothing
        MTLm.draw_primitives!(enc, c.primitive, 0, n, instances)
    else
        ibuf = metal_buffer(indices)
        ibuf === nothing &&
            error("an indexed draw needs a device buffer of indices, got $(typeof(indices))")
        # Already an `MTLRenderStages`; wrapping it is a `MethodError`.
        MTLm.use!(enc, ibuf, MTLm.ReadUsage, MTLm.MTLRenderStageVertex)
        MTLm.draw_indexed_primitives!(enc, c.primitive, n,
                                      MTLm.MTLIndexTypeUInt32, ibuf,
                                      byteoffset(indices), instances)
    end
    return nothing
end

draw_with_count!(enc, c::MetalCompiledGraphicsPipeline, n::Mantle.Commands,
                 ::Integer = 1, ::Nothing = nothing) =
    draw_indirect_with!(enc, c, n)

function draw_indirect_with!(enc, c::MetalCompiledGraphicsPipeline, n::Mantle.Commands)
    # The host never reads the count.
    buf = metal_buffer(n.resource)
    buf === nothing && error("a Commands draw needs a device buffer, got $(typeof(n.resource))")
    # Must be made resident although no shader names it: a non-resident indirect
    # buffer yields a garbage vertex count and a command buffer that never completes.
    MTLm.use!(enc, buf, MTLm.ReadUsage,
              MTLm.MTLRenderStages(MTLm.MTLRenderStageVertex |
                                   MTLm.MTLRenderStageFragment))
    # At the resource's offset: transients are views into the arena.
    MTLm.draw_primitives_indirect!(enc, c.primitive, buf, byteoffset(n.resource))
end

"""The `MTLTexture` behind a render attachment."""
metal_texture(t::MTLm.MTLTexture) = t
metal_texture(t::MetalTexture2D)  = t.tex
metal_texture(t) = error("not something Metal can attach as a render target: $(typeof(t))")

"""The `MTLBuffer` behind a draw argument, or `nothing` if it is not one."""
metal_buffer(x::MTLm.MTLBuffer) = x
# `unsafe_convert` also marks the array as used by this task's next submission, so
# `commit!` claims it and a host read waits for the pass. For a view over a pool
# region this is the block's buffer.
metal_buffer(x::Metal.MtlArray) = Base.unsafe_convert(MTLm.MTLBuffer, x)
metal_buffer(@nospecialize(x))  = nothing

# `Clear` is matched first: `discards` is also true for `Clear`, which would map
# it to `DontCare`.
mtl_loadaction(::Mantle.Clear)   = MTLm.MTLLoadActionClear
mtl_loadaction(l::Mantle.LoadOp) = Mantle.discards(l) ? MTLm.MTLLoadActionDontCare :
                                                        MTLm.MTLLoadActionLoad
mtl_loadaction(::Nothing) = MTLm.MTLLoadActionDontCare

# KernelInterface shader builtins as `@device_override`s in Metal's method table;
# the host methods error and must stay.
const METAL_BUILTINS = (:vertex_index, :instance_index, :frag_coord_x, :frag_coord_y,
                        :frag_coord_z, :frag_coord_w, :frag_coord_xy)

for f in METAL_BUILTINS
    @eval Metal.@device_override KI.$f() = Metal.$f()
end
Metal.@device_override KI.frag_coord(dim::Integer = 1) = Metal.frag_coord(dim)

# `air.dfdx.f32`/`air.dfdy.f32` (`Metal/src/device/intrinsics/graphics.jl`).
Metal.@device_override KI.dFdx(v::Float32) = Metal.dfdx(v)
Metal.@device_override KI.dFdy(v::Float32) = Metal.dfdy(v)

# `air.discard_fragment`: returns and does not terminate the shader, as
# `KI.discard` specifies.
Metal.@device_override KI.discard() = Metal.discard_fragment()

# `[[point_size]]`, appended to the vertex output by `stage_return!` when called.
Metal.@device_override KI.set_point_size!(s::Float32) = Metal.set_point_size!(s)

# `emit_vertex!`, `end_primitive!` and `primitive_id_in` have no Metal
# counterpart: Metal has no geometry stage, and `Mantle.lower_geometry_to_mesh`
# never emits them (`supports_geometry_stage`).

# The binding is a slot; the lowering in `Metal/src/compiler/texture.jl` maps it to
# an entry parameter. `Mantle.FragmentShader`'s `textures` sets how many exist.
Metal.@device_override KI.sample_texture_2d(binding::UInt32, u::Float32, v::Float32,
                                            component::UInt32) =
    Metal.sample_texture_2d(binding, u, v, component)

# Metal's clip-space +y is up, Vulkan's (the portable one) down. An overlay, not
# a method redefinition, so the host answer is unchanged; it applies to compute
# too, so compute reprojection matches the rasteriser. A plain `@overlay`, not
# `@device_override` (a `@consistent_overlay`): Julia would constant-fold calls
# with the host method, which returns `y` unchanged.
Base.Experimental.@overlay Metal.method_table KI.clip_y(y::Float32) = -y

@inline flip_clip(p::Vec4f) = Vec4f(p[1], KI.clip_y(p[2]), p[3], p[4])
@inline flip_clip(p::NTuple{4,Float32}) = (p[1], KI.clip_y(p[2]), p[3], p[4])

"""
The bound texture table on Metal: textures and samplers as lists, since Metal
has no descriptor set; `use_bindings!` sets them on the encoder.
"""
struct MetalTextureBindings <: Mantle.TextureBindings
    textures::Vector{MTLm.MTLTexture}
    samplers::Vector{MTLm.MTLSamplerState}
end

Base.length(b::MetalTextureBindings) = length(b.textures)

# Dispatches on the texture type, since `bind_textures` takes no device.
function Mantle.bind_textures(
        textures::Vector{<:Mantle.SampledTexture{<:Any,<:Any,<:MetalTexture2D}})
    isempty(textures) && error("bind_textures: cannot bind an empty texture list")
    return MetalTextureBindings([st.texture.tex for st in textures],
                                [st.sampler.state for st in textures])
end

"""
    use_bindings!(emitter, compiled, bindings)

Put `bindings` on the encoder for the next draw. Slots are 1-based; the setters
convert to Metal's 0-based slots.
"""
function Mantle.use_bindings!(h::MetalPassHandle, _compiled,
                              b::MetalTextureBindings)
    enc = h.encoder
    for (i, tex) in enumerate(b.textures)
        MTLm.set_fragment_texture!(enc, tex, i)
    end
    for (i, st) in enumerate(b.samplers)
        MTLm.set_fragment_sampler!(enc, st, i)
    end
    return nothing
end
