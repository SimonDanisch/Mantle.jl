# Rasterisation on Metal: the resources behind Mantle's `Framebuffer`,
# `Texture2D`, `Sampler` and `CompiledGraphicsPipeline`, pipeline state
# translation, and the render-pass verbs. The graph decides what is drawn and in
# what order; this records it.

# One method per state value, not a `Dict`, so an unhandled value is a
# `MethodError`, not a `KeyError` mid-frame.

mtl_cull(::Mantle.NoCull)    = MTL.MTLCullModeNone
mtl_cull(::Mantle.CullBack)  = MTL.MTLCullModeBack
mtl_cull(::Mantle.CullFront) = MTL.MTLCullModeFront

mtl_primitive(::Mantle.TriangleList)  = MTL.MTLPrimitiveTypeTriangle
mtl_primitive(::Mantle.TriangleStrip) = MTL.MTLPrimitiveTypeTriangleStrip
mtl_primitive(::Mantle.LineList)      = MTL.MTLPrimitiveTypeLine
mtl_primitive(::Mantle.PointList)     = MTL.MTLPrimitiveTypePoint

mtl_depth_compare(mode::Mantle.DepthMode) = getproperty(
    (less = MTL.MTLCompareFunctionLess, lessequal = MTL.MTLCompareFunctionLessEqual,
     greater = MTL.MTLCompareFunctionGreater, always = MTL.MTLCompareFunctionAlways),
    something(Mantle.depthtest(mode), :always))

"""Configure one colour attachment's blending from the portable `BlendMode`."""
function apply_blend!(att, mode::Mantle.BlendMode)
    Mantle.writescolor(mode) || (att.writeMask = MTL.MTLColorWriteMaskNone)
    f = Mantle.blendfactors(mode)
    att.blendingEnabled = f !== nothing
    f === nothing && return att
    mtl = (one = MTL.MTLBlendFactorOne, srcalpha = MTL.MTLBlendFactorSourceAlpha,
           oneminussrcalpha = MTL.MTLBlendFactorOneMinusSourceAlpha)
    att.sourceRGBBlendFactor, att.destinationRGBBlendFactor,
        att.sourceAlphaBlendFactor, att.destinationAlphaBlendFactor = map(s -> getproperty(mtl, s), f)
    return att
end

function apply_attachments!(desc, color_formats, depth_format, blend)
    for (i, fmt) in enumerate(color_formats)
        att = desc.colorAttachments[i]
        att.pixelFormat = fmt
        apply_blend!(att, Mantle.attachmentblend(blend, i,
                     fmt in (MTL.MTLPixelFormatR32Uint, MTL.MTLPixelFormatRG32Uint)))
    end
    depth_format === nothing || (desc.depthAttachmentPixelFormat = depth_format)
    return desc
end

function depthstate(dev, depth, depth_format)
    (depth_format === nothing || depth isa Mantle.DepthOff) && return nothing
    dd = MTL.MTLDepthStencilDescriptor()
    dd.depthCompareFunction = mtl_depth_compare(depth)
    dd.depthWriteEnabled = Mantle.depthwrites(depth)
    return MTL.MTLDepthStencilState(dev, dd)
end

"""A 2D texture on Metal. The element type is Mantle's spelling of the format."""
struct MetalTexture2D{T} <: Mantle.Texture2D{T}
    tex::MTL.MTLTexture
    width::Int
    height::Int
    dev::MetalDevice
end

Base.size(t::MetalTexture2D) = (t.width, t.height)
Mantle.mipmapped(t::MetalTexture2D) = t.tex.mipmapLevelCount > 1

"""A sampler on Metal."""
struct MetalSampler <: Mantle.Sampler
    state::MTL.MTLSamplerState
end

# Core's `Texture2D` (`graphics/resources.jl`) chose the format and the mip levels.
# `Shared` storage, not a buffer-backed linear texture.
function maketexture(d::MetalDevice, ::Type{T}, format::MTL.MTLPixelFormat, w::Int, h::Int,
                     levels::Int) where {T}
    desc = MTL.MTLTextureDescriptor(format, w, h, levels > 1)
    desc.mipmapLevelCount = levels
    desc.usage = MTL.MTLTextureUsageShaderRead
    desc.storageMode = MTL.MTLStorageModeShared
    return MetalTexture2D{T}(MTL.MTLTexture(d.dev, desc), w, h, d)
end

"""
    copytexels!(tex::MetalTexture2D, data)

New texels for an existing texture, from the host or the device, through a
blit on Mantle's queue rather than `replace_region!`, so a frame still sampling
the texture is not overwritten. The matrix's column-major bytes are uploaded as
they are: one Julia column is one image row. Waits for completion.
"""
function Mantle.copytexels!(t::MetalTexture2D{T}, data::AbstractMatrix{T}) where {T}
    # `collect` for anything without a dense host pointer (views, adjoints).
    rows = data isa Matrix{T} ? data : collect(data)
    nbytes = length(rows) * sizeof(T)
    staging = GC.@preserve rows MTL.MTLBuffer(t.dev.dev, nbytes, pointer(rows);
                                               storage = Metal.SharedStorage)
    blit_into_texture!(t, staging, 0)
    return t
end

# Device source: blit from the array's own buffer. A separate method because an
# `MtlArray` is a `DenseMatrix` whose `pointer` is a device address.
function Mantle.copytexels!(t::MetalTexture2D{T}, data::Metal.MtlMatrix{T}) where {T}
    # The kernel that wrote `data` may be on this task's own Metal.jl queue (no
    # `adopt!`), which `framebuffer!` does not order against.
    Metal.synchronize()
    p = pointer(data)
    blit_into_texture!(t, p.buffer, p.offset)
    return t
end

"""Copy a whole texture's worth of texels from `src` at byte `soff`, on the queue."""
function blit_into_texture!(t::MetalTexture2D{T}, src::MTL.MTLBuffer, soff::Integer) where {T}
    w, h = size(t)
    # Behind the device's uncommitted batch: commit order is execution order.
    cb = framebuffer!(t.dev)
    MTL.MTLBlitCommandEncoder(cb) do enc
        MTL.append_copy!(enc, t.tex, MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(w, h, 1),
                          src, soff, w * sizeof(T), 0)
        t.tex.mipmapLevelCount > 1 && MTL.append_generate_mipmaps!(enc, t.tex)
    end
    # Waited for, so the caller may reuse its array on return.
    submitwait!(cb)
    cb.status == MTL.MTLCommandBufferStatusCompleted ||
        error("Metal texture upload failed: $(cb.status)")
    return nothing
end

# Core's `Sampler` (`graphics/resources.jl`) checked the keywords. Mip levels blend
# linearly and the border is transparent black, as on Vulkan.
function makesampler(d::MetalDevice, filter::Symbol, wrap::Symbol, anisotropy::Float32)
    f = filter === :nearest ? MTL.MTLSamplerMinMagFilterNearest : MTL.MTLSamplerMinMagFilterLinear
    a = wrap === :repeat ? MTL.MTLSamplerAddressModeRepeat :
        wrap === :clamp  ? MTL.MTLSamplerAddressModeClampToEdge :
        wrap === :mirror ? MTL.MTLSamplerAddressModeMirrorRepeat :
                           MTL.MTLSamplerAddressModeClampToBorderColor
    desc = MTL.MTLSamplerDescriptor()
    desc.minFilter = f
    desc.magFilter = f
    desc.mipFilter = MTL.MTLSamplerMipFilterLinear
    desc.sAddressMode = a
    desc.tAddressMode = a
    desc.borderColor = MTL.MTLSamplerBorderColorTransparentBlack
    # Metal counts the one sample of no anisotropy as 1.
    desc.maxAnisotropy = max(1, round(Int, anisotropy))
    desc.normalizedCoordinates = true
    return MetalSampler(MTL.MTLSamplerState(d.dev, desc))
end

"""
An offscreen render target: a `Shared` colour texture and an optional depth one.
Not buffer-backed: rendering into a linear texture on an Apple GPU silently writes
nothing, so readback uses `getBytes!`.
"""
struct MetalFramebuffer <: Mantle.Framebuffer
    color::MTL.MTLTexture
    depth::Union{Nothing,MTL.MTLTexture}
    width::Int
    height::Int
    color_format::MTL.MTLPixelFormat
    # Its passes commit on this device's queue, which a readback waits for.
    dev::MetalDevice
end

# Core's `Framebuffer` (`graphics/resources.jl`) chose the formats.
function makeframebuffer(d::MetalDevice, width::Int, height::Int, cfmt::MTL.MTLPixelFormat,
                         dfmt::Union{Nothing,MTL.MTLPixelFormat})
    dev = d.dev
    cdesc = MTL.MTLTextureDescriptor(cfmt, width, height, false)
    cdesc.usage = MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead
    cdesc.storageMode = MTL.MTLStorageModeShared
    ctex = MTL.MTLTexture(dev, cdesc)
    dtex = nothing
    if dfmt !== nothing
        ddesc = MTL.MTLTextureDescriptor(dfmt, width, height, false)
        ddesc.usage = MTL.MTLTextureUsageRenderTarget
        # Private: depth is never read back.
        ddesc.storageMode = MTL.MTLStorageModePrivate
        dtex = MTL.MTLTexture(dev, ddesc)
    end
    return MetalFramebuffer(ctex, dtex, width, height, cfmt, d)
end

# Metal's value for each of core's pixel layouts (`runtime/format.jl`). No
# three-channel formats exist on Metal; callers pad to four.
nativeformat(::MetalAPI, ::Type{BGRA{N0f8}}, srgb::Bool) =
    srgb ? MTL.MTLPixelFormatBGRA8Unorm_sRGB : MTL.MTLPixelFormatBGRA8Unorm
nativeformat(::MetalAPI, ::Type{RGBA{N0f8}}, srgb::Bool) =
    srgb ? MTL.MTLPixelFormatRGBA8Unorm_sRGB : MTL.MTLPixelFormatRGBA8Unorm
nativeformat(::MetalAPI, ::Type{RGBA{Float16}}, ::Bool) = MTL.MTLPixelFormatRGBA16Float
nativeformat(::MetalAPI, ::Type{RGBA{Float32}}, ::Bool) = MTL.MTLPixelFormatRGBA32Float
nativeformat(::MetalAPI, ::Type{Depth32}, ::Bool) = MTL.MTLPixelFormatDepth32Float
nativeformat(::MetalAPI, ::Type{Float32}, ::Bool) = MTL.MTLPixelFormatR32Float
nativeformat(::MetalAPI, ::Type{Float16}, ::Bool) = MTL.MTLPixelFormatR16Float
nativeformat(::MetalAPI, ::Type{UInt32}, ::Bool) = MTL.MTLPixelFormatR32Uint
nativeformat(::MetalAPI, ::Type{SVector{2,UInt32}}, ::Bool) = MTL.MTLPixelFormatRG32Uint
nativeformat(::MetalAPI, ::Type{NTuple{2,Float32}}, ::Bool) = MTL.MTLPixelFormatRG32Float
nativeformat(::MetalAPI, ::Type{NTuple{2,Float16}}, ::Bool) = MTL.MTLPixelFormatRG16Float
nativeformat(::MetalAPI, ::Type{UInt8}, ::Bool) = MTL.MTLPixelFormatR8Unorm

"""A compiled Metal render pipeline, plus the state a draw still has to set."""
struct MetalCompiledGraphicsPipeline <: Mantle.CompiledGraphicsPipeline
    state::MTL.MTLRenderPipelineState
    depth_state::Union{Nothing,MTL.MTLDepthStencilState}
    primitive::MTL.MTLPrimitiveType
    cull::MTL.MTLCullMode
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
fragment_attachment_type(fmt) = fmt == MTL.MTLPixelFormatR32Uint ? UInt32 :
    fmt == MTL.MTLPixelFormatRG32Uint ? NTuple{2,UInt32} : NTuple{4,Float32}
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
                          color_formats::Vector{MTL.MTLPixelFormat},
                          depth_format::Union{Nothing,MTL.MTLPixelFormat},
                          vert_bufs::Type, frag_bufs::Type)
    # Geometry pipelines normally go through `compile_draw`, which lowers them
    # onto a mesh pipeline (`metal/mesh.jl`); this compiles native stages only.
    p.geometry === nothing || Mantle.supports_geometry_stage(d) ||
        error("`compile_pipeline` compiles each stage of the pipeline as itself, " *
              "and this backend has no geometry stage to compile one into. Apple's " *
              "replacement is the mesh pipeline, which this backend DOES run: go " *
              "through `Mantle.compile_draw`, which lowers a geometry pipeline " *
              "with `Mantle.lower_geometry_to_mesh` and compiles the mesh " *
              "pipeline that comes out. Ask `supports_geometry_stage(backend)` " *
              "only if you need to know which of the two ran.")

    dev = d.dev
    vfn, ffn, vert_tt, frag_tt =
        stage_signatures(p, length(color_formats), vert_bufs, frag_bufs; color_formats)
    vstage = compile_stage_function(dev, vfn, vert_tt, :vertex,
                                    stagename(Mantle.stagefunction(p.vertex), "_vs"))
    fstage = ffn === nothing ? nothing :
        compile_stage_function(dev, ffn, frag_tt, :fragment,
                               stagename(Mantle.stagefunction(p.fragment), "_fs"))

    # Native stages carry compiled code identity, independent of captured data.
    key = pipelinekey((vstage, fstage), color_formats, depth_format, p)
    return get!(d.pipelines, key) do
        vfun, vlib = vstage
        ffun, flib = fstage === nothing ? (nothing, nothing) : fstage

        desc = MTL.MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfun
        ffun === nothing || (desc.fragmentFunction = ffun)
        # Attachments in pass order, matching `air.render_target <i>`.
        apply_attachments!(desc, color_formats, depth_format, p.blend)
        state = MTL.MTLRenderPipelineState(dev, desc)
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
function compile_stage_function(dev::MTL.MTLDevice, f, tt::Type, stage::Symbol, name::String)
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
    lib = MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib)
    compiled = (MTL.MTLFunction(lib, name), lib)
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
framebuffer!(::MTL.MTLDevice) = framebuffer!(Device(MetalAPI()))

function framebuffer!(d::MetalDevice)
    bq = batchqueue(d)
    Metal.has_open_batch(bq) && Metal.flush!(bq)
    return MTL.MTLCommandBuffer(bq.queue)
end

"""Commit a command buffer from `framebuffer!`; every caller ends with this or
`submitwait!`. `Metal.submit_command_buffer` commits Metal.jl's open batch first
and claims the bound arrays (`metal_buffer`) for this queue, so host reads of
them wait for this buffer."""
function commit!(cb::MTL.MTLCommandBuffer)
    Metal.submit_command_buffer(() -> MTL.commit!(cb), cb, true)
    committed!(cb)
    return cb
end

"""Commit and wait; required before reading bytes back."""
function submitwait!(cb::MTL.MTLCommandBuffer)
    commit!(cb)
    MTL.wait_completed(cb)
    return nothing
end

# Per-pass GPU time from each command buffer's `GPUStartTime`/`GPUEndTime`
# (seconds). Committed buffers are collected while profiling and drained by
# `gpupasstime!`, which submits and waits at every pass boundary.
const COMMITTED = MTL.MTLCommandBuffer[]

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
    open_cb === nothing || push!(COMMITTED, MTL.retained(open_cb))
    Metal.flush!(bq)
    isempty(COMMITTED) && return 0.0
    total = 0.0
    for cb in COMMITTED
        MTL.wait_completed(cb)
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
    # `getBytes!` is a host copy that does not wait for the device's queue.
    waitidle(fb.dev)
    # The colorant's layout matches the texel's, so bytes are read directly.
    px = Matrix{eltypeof(fb.color_format)}(undef, fb.width, fb.height)
    GC.@preserve px MTL.getBytes!(pointer(px), fb.color, fb.width * sizeof(eltype(px)),
                                   MTL.MTLRegion(MTL.MTLOrigin(0, 0, 0),
                                                  MTL.MTLSize(fb.width, fb.height, 1)))
    return px
end

supports_graphics(::MetalDevice) = true

# The render-pass verbs declared in `graphics/commands.jl`, driven by core's
# `runrenderpass!`.

"""What `begin_render_pass!` returns and `record_draw!`/`end_render_pass!` take."""
struct MetalPassHandle
    cmdbuf::MTL.MTLCommandBuffer
    encoder::MTL.MTLRenderCommandEncoder
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
    buffers::Vector{MTL.MTLBuffer}
    # Reused staging for `setVertexBytes`, which takes a pointer; avoids a `Ref`
    # per argument per draw.
    scratch::Vector{UInt8}
end

bakearg(a) = Metal.mtlconvert(a)

"""What a baked argument is once it reaches the shader."""
argdevtype(a) = typeof(a)

function StageArgs(args)
    t = Tuple(args)
    bufs = MTL.MTLBuffer[]
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
"""The attachments' native formats as the vector a pipeline descriptor takes."""
mtlformats(color_formats, depth_format) =
    (MTL.MTLPixelFormat[f for f in color_formats], depth_format)

function Mantle.compile_draw(d::MetalDevice, p::Mantle.GraphicsPipeline,
                             color_formats, depth_format, vert_args, frag_args;
                             bindings = nothing)
    vert, frag = StageArgs(vert_args), StageArgs(frag_args)
    cfmts, dfmt = mtlformats(color_formats, depth_format)
    pipeline = compile_pipeline(d, p, cfmts, dfmt, buffer_types(vert), buffer_types(frag))
    return MetalCompiledDraw(pipeline, vert, frag)
end

"""Re-bake every stage that has arguments from this frame's resolved `args`."""
rebakestages!(args, stages::StageArgs...) =
    foreach(s -> isempty(s.device) || rebake!(s, args), stages)

"""Set the pipeline state a draw needs. Front faces are counter-clockwise:
`flip_clip` mirrors y, reversing every primitive's winding, and back-face culling
would otherwise drop the front faces."""
function setdrawstate!(enc, c)
    MTL.set_pipeline!(enc, c.state)
    c.depth_state === nothing || MTL.set_depth_stencil_state!(enc, c.depth_state)
    MTL.set_cull_mode!(enc, c.cull)
    MTL.set_front_facing_winding!(enc, MTL.MTLWindingCounterClockwise)
    return nothing
end

function Mantle.begin_render_pass!(d::MetalDevice, targets, loads, depth, depth_load)
    rp = MTL.MTLRenderPassDescriptor()
    for (i, t) in enumerate(targets)
        att = rp.colorAttachments[i]
        att.texture     = metal_texture(t)
        att.loadAction  = mtl_loadaction(loads[i])
        att.storeAction = MTL.MTLStoreActionStore
        cv = Mantle.clearvalue(loads[i])
        cv === nothing || (att.clearColor = MTL.MTLClearColor(cv[1], cv[2], cv[3], cv[4]))
    end
    if depth !== nothing
        da = rp.depthAttachment
        da.texture     = metal_texture(depth)
        da.loadAction  = mtl_loadaction(depth_load)
        da.storeAction = MTL.MTLStoreActionStore
        dv = depth_load === nothing ? nothing : Mantle.depthclear(depth_load)
        dv === nothing || (da.clearDepth = Float64(dv))
    end
    # This pass's own command buffer; `end_render_pass!` commits it.
    cb  = framebuffer!(d.dev)
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    return MetalPassHandle(cb, enc)
end

"""
Bind one stage's baked arguments with `setBytes:` (a stage takes the
`MtlDeviceArray` struct, not the data), after making every buffer behind them
resident.
"""
function bind_stage!(enc, a::StageArgs, setbytes!, stage::MTL.MTLRenderStages)
    for b in a.buffers
        MTL.use!(enc, b, MTL.ReadUsage, stage)
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
    # `args` is this frame's resolved list, shared by both stages.
    rebakestages!(args, d.vert, d.frag)
    c = d.pipeline
    setdrawstate!(h.encoder, c)
    bind_stage!(h.encoder, d.vert, MTL.set_vertex_bytes!, MTL.MTLRenderStageVertex)
    # Metal rejects fragment bindings on a depth-only pipeline.
    c.has_fragment &&
        bind_stage!(h.encoder, d.frag, MTL.set_fragment_bytes!, MTL.MTLRenderStageFragment)
    draw_with_count!(h.encoder, c, count, instances, indices)
    return nothing
end

"""
    setviewport!(handle, x, y, width, height)

The viewport for the following draws, and a scissor over the same rectangle.
A negative `height` (Vulkan's flipped viewport, `y` naming the bottom edge) is
passed to `MTLViewport` unchanged, which interprets it the same way and undoes
the vertex stage's `flip_clip`. The scissor is core's `scissorrect`, at least a
pixel wide, since Metal refuses an empty one.
"""
function Mantle.setviewport!(h::MetalPassHandle, x::Real, y::Real, w::Real, hgt::Real)
    MTL.set_viewport!(h.encoder,
        MTL.MTLViewport(Float64(x), Float64(y), Float64(w), Float64(hgt), 0.0, 1.0))
    sx, sy, sw, sh = Mantle.scissorrect(x, y, w, hgt)
    MTL.set_scissor!(h.encoder, MTL.MTLScissorRect(UInt(sx), UInt(sy), UInt(max(1, sw)), UInt(max(1, sh))))
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
Mantle.devicearray(d::MetalDevice, data::AbstractArray) = buffercopy(d, data)

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
Mantle.depthformat(fb::MetalFramebuffer) = fb.depth.pixelFormat

Mantle.target_extent(fb::MetalFramebuffer) = (fb.width, fb.height)
Mantle.target_format(fb::MetalFramebuffer) = fb.color_format
# The inverse of `pixelformat` over a framebuffer's colour formats, for an
# attachment's pixel type and for `readback_framebuffer`.
eltypeof(f::MTL.MTLPixelFormat) = eltypefor(MetalAPI(), f)

"""
End the encoder and commit the pass's command buffer.
"""
function Mantle.end_render_pass!(h::MetalPassHandle)
    MTL.endEncoding!(h.encoder)
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
        MTL.draw_primitives!(enc, c.primitive, 0, n, instances)
    else
        ibuf = useindices!(enc, indices, MTL.MTLRenderStageVertex)
        MTL.draw_indexed_primitives!(enc, c.primitive, n,
                                      MTL.MTLIndexTypeUInt32, ibuf,
                                      byteoffset(indices), instances)
    end
    return nothing
end

"""The `MTLBuffer` of `indices`, made resident for `stage`, which reads them although
no shader names them; a non-resident buffer reads as zeros."""
function useindices!(enc, indices, stage)
    ibuf = metal_buffer(indices)
    ibuf === nothing && error("an indexed draw needs a device buffer of indices, got $(typeof(indices))")
    # `stage` is already an `MTLRenderStages`; wrapping it is a `MethodError`.
    MTL.use!(enc, ibuf, MTL.ReadUsage, stage)
    return ibuf
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
    MTL.use!(enc, buf, MTL.ReadUsage,
              MTL.MTLRenderStages(MTL.MTLRenderStageVertex |
                                   MTL.MTLRenderStageFragment))
    # At the resource's offset: transients are views into the arena.
    MTL.draw_primitives_indirect!(enc, c.primitive, buf, byteoffset(n.resource))
end

"""The `MTLTexture` behind a render attachment."""
metal_texture(t::MTL.MTLTexture) = t
metal_texture(t::MetalTexture2D)  = t.tex
metal_texture(t) = error("not something Metal can attach as a render target: $(typeof(t))")

"""The `MTLBuffer` behind a draw argument, or `nothing` if it is not one."""
metal_buffer(x::MTL.MTLBuffer) = x
# `unsafe_convert` also marks the array as used by this task's next submission, so
# `commit!` claims it and a host read waits for the pass. For a view over a pool
# region this is the block's buffer.
metal_buffer(x::Metal.MtlArray) = Base.unsafe_convert(MTL.MTLBuffer, x)
metal_buffer(@nospecialize(x))  = nothing

# `Clear` is matched first: `discards` is also true for `Clear`, which would map
# it to `DontCare`.
mtl_loadaction(::Mantle.Clear)   = MTL.MTLLoadActionClear
mtl_loadaction(l::Mantle.LoadOp) = Mantle.discards(l) ? MTL.MTLLoadActionDontCare :
                                                        MTL.MTLLoadActionLoad
mtl_loadaction(::Nothing) = MTL.MTLLoadActionDontCare

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
    textures::Vector{MTL.MTLTexture}
    samplers::Vector{MTL.MTLSamplerState}
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
        MTL.set_fragment_texture!(enc, tex, i)
    end
    for (i, st) in enumerate(b.samplers)
        MTL.set_fragment_sampler!(enc, st, i)
    end
    return nothing
end
