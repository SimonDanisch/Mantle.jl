# The rasterisation half of the Metal backend.
#
# `supports_graphics(::MetalBackend)` said `false` for one reason: Metal.jl
# compiled Julia to compute kernels and nothing else. It compiles vertex and
# fragment programs now (`Metal/src/compiler/graphics.jl`), so the reason is
# gone and this is the rest — the concrete resources behind Mantle's abstract
# `Framebuffer` / `Texture2D` / `Sampler` / `CompiledGraphicsPipeline`, the
# translation of the portable pipeline state, and `draw!`.
#
# Nothing here knows about the graph. Mantle decides what is drawn and in what
# order; this records it.

const MTLm = Metal.MTL

# ── portable pipeline state → Metal ──────────────────────────────────────────
#
# One method per state value rather than a `Dict`, so an unhandled one is a
# missing method at compile time instead of a `KeyError` in the middle of a
# frame.

mtl_cull(::Mantle.NoCull)    = MTLm.MTLCullModeNone
mtl_cull(::Mantle.CullBack)  = MTLm.MTLCullModeBack
mtl_cull(::Mantle.CullFront) = MTLm.MTLCullModeFront

mtl_primitive(::Mantle.TriangleList)  = MTLm.MTLPrimitiveTypeTriangle
mtl_primitive(::Mantle.TriangleStrip) = MTLm.MTLPrimitiveTypeTriangleStrip
mtl_primitive(::Mantle.LineList)      = MTLm.MTLPrimitiveTypeLine
mtl_primitive(::Mantle.PointList)     = MTLm.MTLPrimitiveTypePoint

mtl_depth_compare(::Mantle.DepthLess)    = MTLm.MTLCompareFunctionLess
mtl_depth_compare(::Mantle.DepthLessEq)  = MTLm.MTLCompareFunctionLessEqual
mtl_depth_compare(::Mantle.DepthGreater) = MTLm.MTLCompareFunctionGreater
mtl_depth_compare(::Mantle.DepthAlways)  = MTLm.MTLCompareFunctionAlways
mtl_depth_compare(::Mantle.DepthOff)     = MTLm.MTLCompareFunctionAlways

depth_writes(::Mantle.DepthMode)  = true
depth_writes(::Mantle.DepthOff)   = false

"""Configure one colour attachment's blending from the portable `BlendMode`."""
function apply_blend!(att, ::Mantle.Opaque)
    att.blendingEnabled = false
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

# ── resources ────────────────────────────────────────────────────────────────

"""A 2D texture on Metal. The element type is Mantle's spelling of the format."""
struct MetalTexture2D{T} <: Mantle.Texture2D{T}
    tex::MTLm.MTLTexture
    width::Int
    height::Int
end

Base.size(t::MetalTexture2D) = (t.width, t.height)

"""A sampler on Metal."""
struct MetalSampler <: Mantle.Sampler
    state::MTLm.MTLSamplerState
end

"""
The pixel format a SAMPLED texture of element type `T` has.

Not `mtlformat`, and the difference is not cosmetic: that one answers for a
RENDER TARGET, where `Float32` means `Depth32Float`. A single-channel float
texture that is read by a shader — a signed-distance glyph atlas is exactly one —
is `R32Float`, and creating it as depth is rejected by the sampler.

Everything else defers, because for a colour format the two questions have the
same answer.
"""
mtlsampledformat(::Type{Float32}) = MTLm.MTLPixelFormatR32Float
mtlsampledformat(::Type{Float16}) = MTLm.MTLPixelFormatR16Float
mtlsampledformat(@nospecialize(T::Type)) = mtlformat(T)

"""
    Texture2D(backend, data::Matrix)

A sampled 2D texture holding `data`.

Dispatches on the BACKEND, matching `Texture2D(::LavaBackend, …)` on the other
side — with two backends loaded there is nothing else to tell them apart by. The
element type names the format, as everywhere a caller names one in Mantle.

`storageMode` is Shared and the upload goes through `replace_region!` rather than
a buffer-backed texture: a linear texture cannot be sampled on an Apple GPU, and
the same storage cannot be both a render target and a sampled source. This is
the same reason `MetalFramebuffer` reads back through `getBytes!`.
"""
function Mantle.Texture2D(::Metal.MetalBackend, data::AbstractMatrix{T}) where {T}
    dev = Metal.device()
    h, w = size(data)          # (row, col), which is (height, width)
    desc = MTLm.MTLTextureDescriptor(mtlsampledformat(T), w, h, false)
    desc.usage = MTLm.MTLTextureUsageShaderRead
    desc.storageMode = MTLm.MTLStorageModeShared
    tex = MTLm.MTLTexture(dev, desc)
    # Metal wants rows contiguous, and a Julia matrix is COLUMN-major, so the
    # transpose is what makes `bytesPerRow` mean what Metal reads it as. Copying
    # rather than reinterpreting, because a lazy transpose has no pointer.
    rows = collect(transpose(data))
    GC.@preserve rows MTLm.replace_region!(
        tex, MTLm.MTLRegion(MTLm.MTLOrigin(0, 0, 0), MTLm.MTLSize(w, h, 1)), 0,
        convert(Ptr{Cvoid}, pointer(rows)), w * sizeof(T))
    return MetalTexture2D{T}(tex, w, h)
end

"""
    Sampler(backend; filter = :linear, wrap = :repeat)

How a texture is sampled. The same two keywords the Vulkan side takes, because a
caller that had to know which backend it was on would be writing two renderers.
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
    desc.sAddressMode = a
    desc.tAddressMode = a
    desc.normalizedCoordinates = true
    return MetalSampler(MTLm.MTLSamplerState(Metal.device(), desc))
end

"""
An offscreen render target: a colour texture and, optionally, a depth one.

Buffer-backed textures are deliberately NOT used. A render target cannot be a
linear texture on an Apple GPU — it builds, it draws, and the result is never
written — so readback goes through `getBytes!` instead.
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

Allocate an offscreen target.

Dispatches on the BACKEND, matching the documented signature — with two backends
loaded there is nothing else to tell them apart by. `color_format` is a Julia
element type, as everywhere a caller names a format in Mantle; `mtlformat`
lowers it the way `vkformat` does on the other side.
"""
function Mantle.Framebuffer(be::Metal.MetalBackend, width::Integer, height::Integer;
                            depth::Bool = true, color_format = nothing)
    dev = Device(be).dev
    # DELETED in phase 1.7: "naming the Julia type here would need ColorTypes".
    # ColorTypes is in Mantle's [deps]; it is simply never `using`-ed. What the
    # default should be is phase 2.7's question.
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
        # A depth buffer is never read back, so it can live where the GPU wants it.
        ddesc.storageMode = MTLm.MTLStorageModePrivate
        dtex = MTLm.MTLTexture(dev, ddesc)
    end
    return MetalFramebuffer(ctex, dtex, Int(width), Int(height), cfmt, dfmt)
end

"""
    mtlformat(T) -> MTLPixelFormat

Metal's format for a Julia element type. The counterpart to `vkformat`, and the
same table `runtime/format.jl` documents.

Matched on the element TYPE. It used to match on `nameof(T)` and the element's
name, because three comments concluded that ColorTypes was not a Mantle
dependency; it is, in `[deps]`, and `runtime/format.jl` documents
`BGRA{N0f8}` as the portable spelling. The structural version accepted any
foreign type that happened to be called `RGBA`.
"""
function mtlformat(@nospecialize(T::Type); srgb::Bool = false)
    # Depth, matching `vkformat`'s `D32_SFLOAT` and the table in
    # `runtime/format.jl`. It said `R32Float` here once, which is a colour
    # format: the one caller hardcoded `Depth32Float` beside it and so never
    # saw it, but a depth target reaching this by way of its element type would
    # have been created as colour and rejected by the render pass.
    T === Float32 && return MTLm.MTLPixelFormatDepth32Float
    T <: Real && error("no Metal pixel format for the scalar $T")
    # On the TYPES, not on `nameof(T)`. The structural match this replaces
    # accepted any type named `RGBA` whose element was an 8-bit `Normed`,
    # from any package, and it was there because three comments said ColorTypes
    # was not a dependency. It is.
    T === BGRA{N0f8}    && return srgb ? MTLm.MTLPixelFormatBGRA8Unorm_sRGB :
                                         MTLm.MTLPixelFormatBGRA8Unorm
    T === RGBA{N0f8}    && return srgb ? MTLm.MTLPixelFormatRGBA8Unorm_sRGB :
                                         MTLm.MTLPixelFormatRGBA8Unorm
    # No sRGB variant exists for the float formats, and none is needed: sRGB is
    # an encoding for 8-bit channels, and a float target holds linear values.
    T === RGBA{Float16} && return MTLm.MTLPixelFormatRGBA16Float
    T === RGBA{Float32} && return MTLm.MTLPixelFormatRGBA32Float
    error("no Metal pixel format for $T — extend `mtlformat` in metal/graphics.jl")
end

"""A compiled Metal render pipeline, plus the state a draw still has to set."""
struct MetalCompiledGraphicsPipeline <: Mantle.CompiledGraphicsPipeline
    state::MTLm.MTLRenderPipelineState
    depth_state::Union{Nothing,MTLm.MTLDepthStencilState}
    primitive::MTLm.MTLPrimitiveType
    cull::MTLm.MTLCullMode
    # A depth-only pipeline has none, and binding to a stage that does not exist
    # is an error rather than a no-op.
    has_fragment::Bool
    # Held so the functions stay reachable: an `MTLFunction` does not keep its
    # library alive, and a released library takes the pipeline's shaders with it.
    libs::Vector{Any}
end

# ── compiling a portable pipeline ────────────────────────────────────────────

"""
    stage_output_type(pipeline, stage) -> Type

The struct a stage writes.

`Mantle.outputtype` builds it from the stage's own `outputs`, and a `NamedTuple`
TYPE already has `fieldnames` and `fieldtype` — which is exactly what Metal.jl's
`stage_outputs` reads. So the portable declaration lowers with no translation
table at all.

The clip position is prepended there rather than declared, because every vertex
stage has one and a caller that had to remember to list it would eventually
forget.
"""
stage_output_type(p::Mantle.GraphicsPipeline, ::Val{:vertex}, ncolor::Int = 1) =
    Mantle.outputtype(p.vertex)

"""
One field per colour attachment, because that is what a fragment stage writes.

The count comes from the PASS, not from the shader: a deferred g-buffer pass
attaches albedo, material and normal and its fragment returns a three-tuple, and
the same shader compiled for one attachment would silently drop two of them.
`air.render_target <i>` is per field, so the struct IS the attachment list.
"""
stage_output_type(::Mantle.GraphicsPipeline, ::Val{:fragment}, ncolor::Int = 1) =
    NamedTuple{ntuple(i -> Symbol(:color, i), ncolor),
               NTuple{ncolor, NTuple{4,Float32}}}

# ── Shaders that RETURN their outputs ────────────────────────────────────────
#
# The portable spelling: a vertex stage returns `(position = …, varyings…)` and a
# fragment stage takes the varyings as its first argument and returns one value
# per colour attachment. That is how `bench/showcase.jl` is written and what
# Lava's `VertexWrapper`/`FragmentWrapper` translate on the other backend.
#
# AIR wants neither shape — a stage writes through the trailing pointer this
# backend's rewrite turns into a return value, and reads each varying as its own
# tagged parameter. These two callables are that translation, and they are
# `@generated` for the same reason Lava's are: the arity and the field list are
# types, so the whole thing folds away and the compiled stage looks as if the
# caller had written the pointer form by hand.

"""
Bring one value to the type the stage's output struct declares.

A shader is free to write `Vec2f` where the declaration says `NTuple{2,Float32}`
or the other way round, and this is the bridge. The two are the SAME bytes — a
`Vec` is a one-field wrapper around exactly that tuple — so it is a `getfield`,
not a `convert`: `convert` has no method between them and the failure is a
`MethodError` reached from device code, which surfaces as
`jl_f_throw_methoderror` plus a `gc_pool_alloc` in a fragment shader.

It is NOT what carries `position`. That field is `Vec4f` on both backends now,
because a clip position is one; declaring it as the tuple here and converting
was a second spelling for one thing, and nothing needed it — `mangle_varying`
gives `Vec4f` and `NTuple{4,Float32}` the same string, which is all the two
stages link by.
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
Assemble a fragment stage's declared outputs from what the shader returned.

One value per attachment when there are several, and just the value when there
is one — which is genuinely ambiguous, because a colour IS a four-tuple:
`r isa Tuple` reads a single `NTuple{4,Float32}` as four render targets and
throws inside the `NamedTuple` constructor, which surfaces as a fragment stage
that is `noreturn` and draws nothing.

So the test is on the ELEMENT type. A per-target tuple holds colours, and a
colour is not a `Real`; a single colour holds floats. That distinguishes the two
for every attachment count.
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

"""A vertex shader that returns `(position = …, varyings…)`."""
struct MetalVertexStage{F, Out} end


@generated function (::MetalVertexStage{F,Out})(args::Vararg{Any,N}) where {F,Out,N}
    call = Expr(:call, :(F.instance), (:(args[$i]) for i in 1:(N - 1))...)
    # By NAME, not by position: the shader is free to list its varyings in
    # whatever order reads well, and the declaration is what fixes the layout.
    # `position` is the one field with a meaning of its own — every vertex stage
    # has one, and it is the one that has to be brought into this backend's
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
        Base.unsafe_store!(args[$N]::Core.LLVMPtr{Out,1}, Out($vals))
        return nothing
    end
end

"""
A fragment shader that takes the varyings as a NamedTuple and returns its
colours.

The varying markers sit BETWEEN the buffer arguments and the output pointer:
buffers keep the leading positions they were given, so their `air.buffer`
location indices are the slots `record_draw!` binds, and the output stays last
because that is what `stage_return!` takes it to be.
"""
struct MetalFragmentStage{F, VIn, Out} end

@generated function (::MetalFragmentStage{F,VIn,Out})(args::Vararg{Any,N}) where {F,VIn,Out,N}
    nv = fieldcount(VIn)
    nbuf = N - 1 - nv
    ins = Expr(:tuple, (:(args[$(nbuf + i)].value) for i in 1:nv)...)
    call = Expr(:call, :(F.instance), :(VIn($ins)), (:(args[$i]) for i in 1:nbuf)...)
    quote
        Base.@_inline_meta
        r = $call
        Base.unsafe_store!(args[$N]::Core.LLVMPtr{Out,1}, stage_fragment_out(Out, r))
        return nothing
    end
end

"""
The Julia signature the wrapped fragment stage is compiled for.

One `Varying` marker per declared varying, in declaration order — the ORDER is
not what links them (the mangled string is), but a stable order keeps the
argument metadata reproducible.
"""
varying_markers(VIn::Type) =
    Tuple(Metal.Varying{n, fieldtype(VIn, n)} for n in fieldnames(VIn))

"""Compiled pipelines, keyed on everything baked into one."""
const GFX_CACHE = Dict{Any,Any}()
const GFX_CACHE_LOCK = ReentrantLock()

"""
What the fragment stage reads, as a `NamedTuple` type.

`Mantle.fragmentinputtype` answers it, because which stage feeds the rasteriser
depends on which stages the pipeline HAS — the geometry stage when there is one,
the vertex stage otherwise. An empty tuple is a real answer, not a missing one:
a vertex stage that outputs only its clip position is a real pipeline and a
shadow pass is exactly that.
"""
varying_type(p::Mantle.GraphicsPipeline) = Mantle.fragmentinputtype(p)

"""
    stage_signatures(p, ncolor, vert_bufs, frag_bufs) -> (vfn, ffn, vert_tt, frag_tt)

The two callables to compile and the Julia signature for each.

Both stages are ALWAYS wrapped. The portable contract is that a shader returns
its outputs and reads its varyings as a NamedTuple, and the pointer form this
backend compiles is an implementation detail of AIR — a caller who had to know
which spelling a backend wanted would be writing two shaders.
"""
function stage_signatures(p::Mantle.GraphicsPipeline, ncolor::Int,
                          vert_bufs::Type, frag_bufs::Type)
    VIn  = varying_type(p)
    VOut = stage_output_type(p, Val(:vertex))
    vfn = MetalVertexStage{typeof(Mantle.stagefunction(p.vertex)), VOut}()
    vert_tt = Tuple{vert_bufs.parameters..., Core.LLVMPtr{VOut,1}}
    # No colour attachment means no fragment stage at all — a shadow pass writes
    # depth and nothing else, and its `fragment` returns `nothing`. Metal spells
    # that as a pipeline with a nil `fragmentFunction`; compiling a stage that
    # returns an EMPTY struct instead is not a thing AIR has.
    ncolor == 0 && return vfn, nothing, vert_tt, nothing
    FOut = stage_output_type(p, Val(:fragment), ncolor)
    ffn = MetalFragmentStage{typeof(Mantle.stagefunction(p.fragment)), VIn, FOut}()
    frag_tt = Tuple{frag_bufs.parameters..., varying_markers(VIn)...,
                    Core.LLVMPtr{FOut,1}}
    return vfn, ffn, vert_tt, frag_tt
end

"""
    compile_pipeline(pipeline, color_formats, depth_format, vert_bufs, frag_bufs)

Compile a portable `GraphicsPipeline` into a Metal render pipeline state.

`vert_bufs`/`frag_bufs` are Tuple types of the stage's BUFFER arguments only.
Everything else about the signature — the varying inputs, the output pointer —
follows from the pipeline, so there is one place that decides it.

Cached on the shader pair, those argument types and the state, because
everything in that list is baked into the compiled object and nothing else is.
"""
function compile_pipeline(p::Mantle.GraphicsPipeline,
                          color_formats::Vector{MTLm.MTLPixelFormat},
                          depth_format::Union{Nothing,MTLm.MTLPixelFormat},
                          vert_bufs::Type, frag_bufs::Type)
    # Asked through the capability, not asserted here: a caller can now ask
    # `supports_geometry_stage(backend)` BEFORE building a pipeline, which is
    # the point of phase 2.7 — the answer used to be reachable only by
    # compiling one and reading the error.
    be = Metal.MetalBackend()
    p.geometry === nothing || Mantle.supports_geometry_stage(be) ||
        error("this backend has no geometry stage. Apple's replacement is the " *
              "mesh pipeline, which this backend DOES run — see " *
              "`Mantle.MeshPipeline` and `supports_mesh_pipeline`. Lower the " *
              "pipeline with `Mantle.lower_geometry_to_mesh` instead of asking " *
              "for a geometry stage; ask `supports_geometry_stage(backend)` " *
              "first if you need to know which you are on.")
    (p.tess_control === nothing && p.tess_eval === nothing) ||
        Mantle.supports_tessellation(be) ||
        error("this backend has no tessellation; ask " *
              "`supports_tessellation(backend)` before building a pipeline " *
              "with it.")

    # The stages themselves, because each now carries its function, its config
    # and its interface — three things that used to be keyed separately and
    # could disagree about which pipeline they belonged to.
    key = (p.vertex, p.fragment, p.geometry, vert_bufs, frag_bufs,
           color_formats, depth_format,
           typeof(p.blend), typeof(p.cull), typeof(p.topology), typeof(p.depth))
    Base.@lock GFX_CACHE_LOCK begin
        cached = get(GFX_CACHE, key, nothing)
        cached === nothing || return cached::MetalCompiledGraphicsPipeline
    end

    dev = Metal.device()
    vfn, ffn, vert_tt, frag_tt =
        stage_signatures(p, length(color_formats), vert_bufs, frag_bufs)
    vname = string(nameof(Mantle.stagefunction(p.vertex))) * "_vs"
    vfun, vlib = compile_stage_function(vfn, vert_tt, :vertex, vname)
    ffun, flib = ffn === nothing ? (nothing, nothing) :
        compile_stage_function(ffn, frag_tt, :fragment,
                               string(nameof(Mantle.stagefunction(p.fragment))) * "_fs")

    desc = MTLm.MTLRenderPipelineDescriptor()
    desc.vertexFunction = vfun
    ffun === nothing || (desc.fragmentFunction = ffun)
    # One descriptor slot per attachment, in the order the pass declared them,
    # which is the order `air.render_target <i>` numbers the fragment's outputs.
    for (i, fmt) in enumerate(color_formats)
        att = desc.colorAttachments[i]
        att.pixelFormat = fmt
        apply_blend!(att, p.blend)
    end
    depth_format === nothing || (desc.depthAttachmentPixelFormat = depth_format)
    state = MTLm.MTLRenderPipelineState(dev, desc)

    dstate = nothing
    if depth_format !== nothing && !(p.depth isa Mantle.DepthOff)
        dd = MTLm.MTLDepthStencilDescriptor()
        dd.depthCompareFunction = mtl_depth_compare(p.depth)
        dd.depthWriteEnabled    = depth_writes(p.depth)
        dstate = MTLm.MTLDepthStencilState(dev, dd)
    end

    compiled = MetalCompiledGraphicsPipeline(state, dstate, mtl_primitive(p.topology),
                                             mtl_cull(p.cull), ffun !== nothing,
                                             Any[l for l in (vlib, flib) if l !== nothing])
    Base.@lock GFX_CACHE_LOCK begin
        GFX_CACHE[key] = compiled
    end
    return compiled
end

"""
    compile_stage_function(f, tt, stage, name) -> (MTLFunction, MTLLibrary)

Compile one Julia shader into a Metal stage function.

The library is handed back, not discarded: an `MTLFunction` does not keep it
alive, and a released library takes the pipeline's shaders with it — which shows
up as a render that draws nothing rather than as an error.
"""
function compile_stage_function(f, tt::Type, stage::Symbol, name::String)
    dev = Metal.device()
    cfg = Metal.compiler_config(dev; stage, name)
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt), cfg)
    lib = MTLm.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib)
    return (MTLm.MTLFunction(lib, name), lib)
end

# ── drawing ──────────────────────────────────────────────────────────────────

"""
    draw!(backend, pipeline, target::OffscreenTarget, vertex_count; …)

Record one draw into `target`.

The portable entry point, the same one the Vulkan backend implements. Mantle
decides what is drawn; this only records it.

`buffers` is what the shaders read, bound in order from slot 1 — Julia's
counting, converted at the encoder boundary. Metal binds by slot rather than
through a descriptor set, so there is nothing between the caller's list and the
stage's arguments.

`vert_tt`/`frag_tt` are the Tuple types of those buffer arguments as the SHADER
sees them, which a raw `MTLBuffer` cannot tell you — it is bytes, and the
element type is the caller's to name. The graph path does not need them: it
resolves real arrays and reads the types off those.
"""
function Mantle.draw!(be::Metal.MetalBackend, p::Mantle.GraphicsPipeline,
                      target::Mantle.OffscreenTarget, vertex_count::Integer;
                      buffers = (), frag_buffers = (), instances::Integer = 1,
                      clear_color::Union{Nothing,NTuple{4,Float32}} = (0f0, 0f0, 0f0, 1f0),
                      depth_clear::Union{Nothing,Float32} = 1f0,
                      vert_tt::Type = Tuple{}, frag_tt::Type = Tuple{})
    fb = target.fb
    fb isa MetalFramebuffer ||
        error("this backend draws into a MetalFramebuffer, got $(typeof(fb))")

    compiled = compile_pipeline(p, MTLm.MTLPixelFormat[fb.color_format],
                                fb.depth === nothing ? nothing : fb.depth_format,
                                vert_tt, frag_tt)

    dev = Metal.device()
    rp = MTLm.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture     = fb.color
    ca.loadAction  = clear_color === nothing ? MTLm.MTLLoadActionLoad :
                                               MTLm.MTLLoadActionClear
    ca.storeAction = MTLm.MTLStoreActionStore
    clear_color === nothing ||
        (ca.clearColor = MTLm.MTLClearColor(clear_color[1], clear_color[2],
                                            clear_color[3], clear_color[4]))
    if fb.depth !== nothing
        da = rp.depthAttachment
        da.texture     = fb.depth
        da.loadAction  = depth_clear === nothing ? MTLm.MTLLoadActionLoad :
                                                   MTLm.MTLLoadActionClear
        da.storeAction = MTLm.MTLStoreActionStore
        depth_clear === nothing || (da.clearDepth = Float64(depth_clear))
    end

    # The immediate path draws and waits: its own command buffer, committed and
    # waited for at the end.
    cb = framebuffer!(dev)
    enc = MTLm.MTLRenderCommandEncoder(cb, rp)
    MTLm.set_pipeline!(enc, compiled.state)
    compiled.depth_state === nothing ||
        MTLm.set_depth_stencil_state!(enc, compiled.depth_state)
    MTLm.set_cull_mode!(enc, compiled.cull)
    for (i, b) in enumerate(buffers)
        MTLm.set_vertex_buffer!(enc, b, 0, i)
    end
    for (i, b) in enumerate(frag_buffers)
        MTLm.set_fragment_buffer!(enc, b, 0, i)
    end
    MTLm.draw_primitives!(enc, compiled.primitive, 0, vertex_count, instances)
    MTLm.endEncoding!(enc)
    submitwait!(cb)
    return nothing
end

# ── How this backend submits ─────────────────────────────────────────────────
#
# **One queue, shared with compute**, and one command buffer PER CALL: a
# render pass, a copy, a present, a readback each open their own, and commit
# it before the call that opened it returns.
#
# The queue first, because it is a correctness matter. A second queue of its own
# is what this had, and it quietly threw away the ordering the graph had just
# computed: two Metal queues have NO order between them, so a shadow pass
# reading the draw count a compute pass wrote read whatever was in the buffer —
# and an indirect draw whose count is uninitialised memory does not fail, it
# asks the GPU for four billion vertices. The frame hung in
# `waitUntilCompleted` with the process at 0% CPU. Command buffers on ONE queue
# run in COMMIT order, which is exactly the ordering Mantle's barrier phase
# established.
#
# One buffer per call, because the alternative was an open one. This backend
# used to keep a command buffer open across consecutive render and copy passes
# and commit it only where the graph said a dispatch was about to follow
# (`Mantle.submit!(device)`): that is the open command buffer the Vulkan
# backend deleted, by the same rule — what is open depends on every call since
# it was opened, and nothing can be scheduled around it. A submission costs
# more than the drawing in it (`bench/showcase.jl`'s frame is 5 ms of GPU work
# and took ten times that when every pass committed), so the per-pass
# submissions this costs are the same regression the Vulkan side accepts for
# its ad hoc launches; batching is the graph's job, later, not the queue's.
#
# What stays is the flush of Metal.jl's OWN batched compute before one of our
# buffers is created: Metal.jl accumulates KA dispatches in a command buffer of
# its own, and if one is open, ours has to go behind it.

"""
A fresh command buffer for this backend's own work, behind whatever compute
Metal.jl has batched and not yet committed.
"""
function framebuffer!(dev)
    bq = Metal.global_queue(dev)
    # Compute was recorded since our last buffer. Theirs first, then ours.
    bq.cmdbuf === nothing || Metal.flush!(bq)
    return MTLm.MTLCommandBuffer(bq.queue)
end

"""Commit a command buffer this backend opened. Every caller of `framebuffer!`
ends with this or with `submitwait!`."""
function commit!(cb::MTLm.MTLCommandBuffer)
    MTLm.commit!(cb)
    committed!(cb)
    return cb
end

"""Commit and wait — the only way to read bytes back."""
function submitwait!(cb::MTLm.MTLCommandBuffer)
    commit!(cb)
    MTLm.wait_completed(cb)
    return nothing
end

# ── Per-pass GPU time ────────────────────────────────────────────────────────
#
# A `MTLCommandBuffer` reports `GPUStartTime` and `GPUEndTime` once it has run,
# in seconds. That is a whole profiler with no counter sample buffers and no
# `MTLCounterSet` — but it is per COMMAND BUFFER, so attributing it to a pass
# means one buffer per pass, which means submitting and waiting at every pass
# boundary. `gpupasstime!` does exactly that, and only while profiling.
#
# The buffers this backend commits itself (render, copy, present) are collected
# here as they go; the compute passes are in Metal.jl's batched queue, whose
# open buffer is picked up the same way just before it is flushed.

const COMMITTED = MTLm.MTLCommandBuffer[]

# Armed by `makeprofiler`, and off otherwise. Collecting unconditionally is a
# LEAK: `gpupasstime!` is the only thing that drains this, and it is only called
# while profiling — so an ordinary frame loop pushed ~8 command buffers per
# frame and never let one go. Holding them alive stops Metal recycling them, and
# the frame time climbed with the frame count: 532 ms over the first fifty and
# 1040 ms over the next fifty, for a frame whose GPU work is 5 ms.
const COLLECT = Ref(false)

# How many command buffers may be held before the oldest is let go.
#
# A BOUND rather than trust, because arming is per DEVICE and draining is per
# profiled PLAN: two plans on one device, one of them profiled, and the
# unprofiled one's buffers pile up with nothing to drain them. Unbounded that is
# a leak — holding command buffers alive stops Metal recycling them, and the
# frame time climbs with the frame count until it is ten times what it should
# be. Bounded it is at worst a wrong number for a pass nobody is measuring.
#
# A frame commits under ten, so a profiled run never reaches this.
const COMMITTED_MAX = 64

"""Remember a command buffer so a profiled run can ask what it cost."""
function committed!(cb)
    COLLECT[] || return cb
    length(COMMITTED) >= COMMITTED_MAX && popfirst!(COMMITTED)
    push!(COMMITTED, cb)
    return cb
end

"""
Arm per-pass GPU collection along with the host profiler.

The plan is what knows whether it is profiled, and this is where the backend
finds out. Armed for the DEVICE rather than the plan: two plans on one device
share the queue, so an unprofiled one's buffers are collected too and drained by
the profiled one's next pass. That skews the profiled numbers upward by whatever
the other plan submitted, which is the honest cost of measuring one of two plans
running together.
"""
function Mantle.makeprofiler(dev::MetalDevice, passes, profile::Bool)
    profile || return nothing
    COLLECT[] = true
    return Mantle.hostprofiler(passes)
end

"""
Nanoseconds of GPU time for everything recorded since the last call.

Submits and WAITS, which is what makes the number per-pass and what makes a
profiled frame slower than a real one. A buffer that never reached the GPU
reports zeros for both ends, and contributes nothing rather than a negative.
"""
function Mantle.gpupasstime!(d::MetalDevice)
    bq = Metal.global_queue(d.dev)
    # Every buffer this backend opened for the pass has been committed by the
    # call that opened it; Metal.jl's batch may still be open and is picked up
    # here, so the numbers describe the frame as it ran.
    open_cb = bq.cmdbuf
    open_cb === nothing || push!(COMMITTED, open_cb)
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
    readback_eltype(format) -> Type

The STORAGE type of one channel: `UInt8` for the 8-bit formats, `Float16`/
`Float32` for the float ones.

Not `eltype(eltypeof(format))`, which answers `N0f8` for a `BGRA{N0f8}` — a
normalised type whose value is already in 0..1. A caller that then divides by 255
gets a number 255 times too small, which is what turned a rendered frame into a
nearly-black one with everything still in the right place.
"""
readback_eltype(f::MTLm.MTLPixelFormat) = storagetype(eltype(eltypeof(f)))
storagetype(::Type{T}) where {T <: FixedPoint} = FixedPointNumbers.rawtype(T)
storagetype(::Type{T}) where {T} = T

"""
    readback_framebuffer(fb) -> Matrix{NTuple{4,T}}

The colour attachment on the host: `width` x `height`, one 4-component tuple per
pixel, in the attachment's own channel order.

Through `getBytes!` rather than shared memory: a render target cannot be a
buffer-backed linear texture on an Apple GPU.

The SHAPE is the contract and not this backend's convenience. It returned a
`4 x width x height` `Array{UInt8,3}` while the Vulkan side returned exactly the
matrix above, so `pixels[col, row]` — the one caller, RayMakie's compositor —
read a `BoundsError` here and four correct bytes there. A hook whose two
implementations disagree about the shape of their answer is the thing this
package exists to remove; the divergence only surfaced once a Mac composited an
overlay at all.
"""
function Mantle.readback_framebuffer(fb::MetalFramebuffer)
    # WAITS, because `readback_framebuffer`'s contract says it does: "there is no
    # way to read pixels the device has not finished writing, so unlike the rest
    # of the frame loop this one waits."
    #
    # It did not, and `getBytes!` is a HOST copy out of the texture — it does not
    # join the queue and it does not block on it. So a readback taken straight
    # after a pass read whatever the texture held, which for a freshly created
    # one is zeros. It is a race, so it passed for a long time: RayMakie's
    # compositor happens to synchronise elsewhere in the frame and the window is
    # narrow. `blit!` followed immediately by a readback loses it every time, and
    # the whole composited image came back transparent black.
    Metal.synchronize()
    T = readback_eltype(fb.color_format)
    px = Matrix{NTuple{4,T}}(undef, fb.width, fb.height)
    GC.@preserve px MTLm.getBytes!(pointer(px), fb.color, fb.width * sizeof(eltype(px)),
                                   MTLm.MTLRegion(MTLm.MTLOrigin(0, 0, 0),
                                                  MTLm.MTLSize(fb.width, fb.height, 1)))
    return px
end

# Metal.jl compiles graphics stages now, so the capability answer changes. It
# said `false` in `device.jl` because there was no vertex or fragment program to
# be had, not because Metal cannot rasterise.
supports_graphics(::Metal.MetalBackend) = true
supports_graphics(::MetalDevice) = true

# ── the graph's render-pass verbs ────────────────────────────────────────────
#
# Mantle's `runrenderpass!` decides which pass runs, over which attachments and
# with which draws; these four only do as they are told. Declared in
# `graphics/commands.jl`.

"""What Mantle hands `record_draw!` and `end_render_pass!` back."""
struct MetalPassHandle
    cmdbuf::MTLm.MTLCommandBuffer
    encoder::MTLm.MTLRenderCommandEncoder
end

# ── The argument ABI a graph draw uses ───────────────────────────────────────
#
# The SAME one a kernel launch uses, and deliberately: `mtlconvert` turns an
# `MtlArray` into a `MtlDeviceArray` — a struct of a GPU address and the
# dimensions — and `set_argument!` binds that struct's BYTES to the slot. A
# shader reached through the graph therefore sees exactly what a hand-launched
# kernel's does, which is what lets one shader source compile on both backends:
# Lava's device arrays carry their length too, and a shader written against
# `length(buf)` or a bounds-checked index would not survive a raw pointer here.
#
# The conversion is done ONCE, at bake time, for the reason the KA dispatch path
# bakes its arguments: `mtlconvert` without an encoder makes the buffer
# persistently resident, and that commit is a setup cost, not a per-frame one.

"""
The device-side form of one stage's arguments, plus the buffers behind them.

The buffers are kept because a `MtlDeviceArray` is only an ADDRESS once
converted, and an address the render encoder was never told about is not
resident — Metal reads it as zeros rather than faulting. `record_draw!` hands
each one to `use!`.

Everything is converted once here, because `mtlconvert` on a buffer also makes
it persistently resident and that is a setup cost. A `Ref` used to be kept as a
`Ref` and re-read per frame; `refuserefs` refuses one at the declaration now,
and a value that changes between frames is a `GPURef`.
"""
struct StageArgs
    device::Tuple      # what gets bound, byte for byte
    buffers::Vector{MTLm.MTLBuffer}
end

bakearg(a) = Metal.mtlconvert(a)

"""What a baked argument is once it reaches the shader."""
argdevtype(a) = typeof(a)

function StageArgs(args)
    bufs = MTLm.MTLBuffer[]
    for a in args
        b = metal_buffer(a)
        b === nothing || push!(bufs, b)
    end
    return StageArgs(map(bakearg, Tuple(args)), bufs)
end

"""One draw's compiled pipeline and its baked arguments."""
struct MetalCompiledDraw
    pipeline::MetalCompiledGraphicsPipeline
    vert::StageArgs
    frag::StageArgs
end

"""The Tuple type of one stage's device-side arguments."""
buffer_types(a::StageArgs) = Tuple{map(argdevtype, a.device)...}

# `bindings` is accepted and ignored: a texture is bound to an argument slot
# here, so a pipeline that samples is compiled no differently. Vulkan builds its
# pipeline layout around the descriptor set layout and needs it at compile.
function Mantle.compile_draw(d::MetalDevice, p::Mantle.GraphicsPipeline,
                             color_formats, depth_format, vert_args, frag_args;
                             bindings = nothing)
    cfmts = MTLm.MTLPixelFormat[mtlformat(T) for T in color_formats]
    vert = StageArgs(vert_args)
    frag = StageArgs(frag_args)
    # Through `mtlformat`, not a hardcoded `Depth32Float`: a pipeline compiled
    # for a format the attachment does not have is rejected at draw time, and
    # the attachment's format comes from its element type like every other.
    dfmt = depth_format === nothing ? nothing : mtlformat(depth_format)
    pipeline = compile_pipeline(p, cfmts, dfmt, buffer_types(vert), buffer_types(frag))
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
Bind one stage's baked arguments.

`setBytes:` rather than `setBuffer:`, because what a stage takes is the
`MtlDeviceArray` STRUCT — an address plus the dimensions — and not the data. The
data is reached through the address, which is why every buffer behind an
argument is made resident first.
"""
function bind_stage!(enc, a::StageArgs, setbytes!, stage::MTLm.MTLRenderStages)
    for b in a.buffers
        MTLm.use!(enc, b, MTLm.ReadUsage, stage)
    end
    for (i, arg) in enumerate(a.device)
        bind_arg!(setbytes!, enc, arg, i)
    end
    return nothing
end

# A function barrier: the argument tuple is heterogeneous, so the loop above is
# dynamic whatever happens, and one call per argument keeps the byte copy itself
# concrete.
function bind_arg!(setbytes!, enc, arg::T, i::Int) where {T}
    ref = Base.RefValue(arg)
    GC.@preserve ref begin
        ptr = Base.unsafe_convert(Ptr{T}, ref)
        setbytes!(enc, reinterpret(Ptr{Cvoid}, ptr), sizeof(T), i)
    end
    return nothing
end

function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledDraw, args, count;
                             instances::Integer = 1, indices = nothing)
    # `args` is what Mantle resolved; this backend baked their device form at
    # compile time (see `StageArgs`), so what gets bound comes from `d`.
    c = d.pipeline
    MTLm.set_pipeline!(h.encoder, c.state)
    c.depth_state === nothing ||
        MTLm.set_depth_stencil_state!(h.encoder, c.depth_state)
    MTLm.set_cull_mode!(h.encoder, c.cull)
    # Counter-clockwise is front-facing here, because `flip_clip` mirrors y and
    # a mirror reverses the handedness of every triangle. Without this, back-face
    # culling keeps exactly the faces it used to drop: the g-buffer shows the far
    # side of every solid and a shadow map records the depth of the wrong side —
    # which still LOOKS like a shadow map, and still fills, so the only symptom
    # was the demo's shadow-bias knob having no effect at all.
    MTLm.set_front_facing_winding!(h.encoder, MTLm.MTLWindingCounterClockwise)
    bind_stage!(h.encoder, d.vert, MTLm.set_vertex_bytes!, MTLm.MTLRenderStageVertex)
    # A depth-only pipeline has no fragment stage, so binding to it is not just
    # wasted work — Metal rejects a fragment binding on a pipeline that has none.
    c.has_fragment &&
        bind_stage!(h.encoder, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)
    draw_with_count!(h.encoder, c, count, instances, indices)
    return nothing
end

"""
    setviewport!(handle, x, y, width, height)

The rectangle the following draws land in, and the scissor derived from it.

`abs(height)`, and that is deliberate. Vulkan expresses "clip-space +y at the
top" as a NEGATIVE viewport height; Metal has no such thing — this backend
mirrors y in the vertex stage instead (`flip_clip`, and the matching
`MTLWindingCounterClockwise`, see `mantle-clip-space-is-vulkans`). So the sign
carries no extra instruction here: honouring it would flip a second time and
undo the shader's.

NOT verified against Vulkan side by side. A mirrored scene still looks like a
scene, which is exactly how the clip flip got in unnoticed in the first place;
the check is two renders of the same overlay, one per backend, diffed.
"""
function Mantle.setviewport!(h::MetalPassHandle, x::Real, y::Real, w::Real, hgt::Real)
    ah = abs(hgt)
    MTLm.set_viewport!(h.encoder,
        MTLm.MTLViewport(Float64(x), Float64(y), Float64(w), Float64(ah), 0.0, 1.0))
    MTLm.set_scissor!(h.encoder,
        MTLm.MTLScissorRect(UInt(max(0, floor(Int, x))), UInt(max(0, floor(Int, y))),
                            UInt(max(1, ceil(Int, w))), UInt(max(1, ceil(Int, ah)))))
    return nothing
end

# What a hand-recorded pass reads off a framebuffer and a window. Both are one
# field; they are verbs rather than field access so that core never names a
# backend's struct.
"""
    indexbuffer(::MetalDevice, indices) -> MtlVector{UInt32}

An index buffer on Metal: an ordinary device array.

Overridden rather than left to the core default, which answers a `Mantle.Buffer`
— a pool region with a length, not an array. Both are drawable, but a caller
that keeps its index buffer beside its vertex arrays keeps them in one container,
and the Vulkan side already answers with an array (`alloc_index_buffer`). Two
backends whose `indexbuffer` return different KINDS of thing is a difference a
caller has to know about, which is the thing this package exists to remove.

Nothing extra is asked of the allocation: `drawIndexedPrimitives` takes any
buffer, which is why the usage bits Vulkan needs have no counterpart here.
"""
Mantle.indexbuffer(::MetalDevice, indices::AbstractVector{UInt32}) =
    Metal.MtlVector{UInt32}(indices)

Mantle.colorimage(fb::MetalFramebuffer) = fb.color
Mantle.depthimage(fb::MetalFramebuffer) = fb.depth

Mantle.target_extent(fb::MetalFramebuffer) = (fb.width, fb.height)
Mantle.target_format(fb::MetalFramebuffer) = fb.color_format
Mantle.target_extent(t::Mantle.OffscreenTarget) = Mantle.target_extent(t.fb)
Mantle.target_extent(t::Mantle.WindowTarget) = Mantle.target_extent(t.window)

# The ELEMENT type of the attachment, which is what a pipeline is compiled
# against. `MetalFramebuffer` keeps the MTL format it was created with and not the
# Julia type behind it, so this is the reverse of `mtlformat` — and only over the
# formats a blit target can have, because that is the one caller.
Mantle.blittarget(t::Mantle.OffscreenTarget) = eltypeof(t.fb.color_format)
Mantle.blittarget(t::Mantle.WindowTarget) = eltypeof(Mantle.target_format(t.window))

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

One buffer per pass, committed here: nothing stays open across the call
boundary. It used to stay open for the next render or copy pass to share, and
the graph's `submit!` hook closed it before a dispatch — the open buffer this
backend, like the Vulkan one, no longer has.
"""
function Mantle.end_render_pass!(h::MetalPassHandle)
    MTLm.endEncoding!(h.encoder)
    commit!(h.cmdbuf)
    return nothing
end

"""A plain count draws directly; a `Commands` buffer draws INDIRECTLY."""
# `instances` and `indices` reach here from a HAND-RECORDED pass; the graph passes
# neither. An indexed draw needs the buffer resident for the same reason an
# indirect one does — the command processor reads it, not a stage, so nothing in
# the shader names it and the encoder still has to be told.
function draw_with_count!(enc, c::MetalCompiledGraphicsPipeline, n::Integer,
                          instances::Integer = 1, indices = nothing)
    if indices === nothing
        MTLm.draw_primitives!(enc, c.primitive, 0, n, instances)
    else
        ibuf = metal_buffer(indices)
        ibuf === nothing &&
            error("an indexed draw needs a device buffer of indices, got $(typeof(indices))")
        MTLm.use!(enc, ibuf, MTLm.ReadUsage,
                  MTLm.MTLRenderStages(MTLm.MTLRenderStageVertex))
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
    # The whole point of this form is that the host never learns the count, so
    # reading it here to call the direct form would defeat it.
    buf = metal_buffer(n.resource)
    buf === nothing && error("a Commands draw needs a device buffer, got $(typeof(n.resource))")
    # Resident, like every other buffer this pass reaches. It is easy to miss
    # because nothing in the SHADER names it — the command processor reads it,
    # not a stage — but the encoder still has to be told, and a
    # `drawPrimitives:indirectBuffer:` whose buffer was never made resident does
    # not fail: it reads whatever is mapped, takes the vertex count from it, and
    # asks the GPU for however many that is. Four billion vertices is a command
    # buffer that never completes and a frame that hangs in
    # `waitUntilCompleted` with the process at 0% CPU.
    MTLm.use!(enc, buf, MTLm.ReadUsage,
              MTLm.MTLRenderStages(MTLm.MTLRenderStageVertex |
                                   MTLm.MTLRenderStageFragment))
    # At the resource's own offset: a transient's storage is a view into the
    # arena, and offset zero is whichever tenant was placed first.
    MTLm.draw_primitives_indirect!(enc, c.primitive, buf, byteoffset(n.resource))
end

"""The `MTLTexture` behind a render attachment."""
metal_texture(t::MTLm.MTLTexture) = t
metal_texture(t::MetalTexture2D)  = t.tex
metal_texture(t) = error("not something Metal can attach as a render target: $(typeof(t))")

"""The `MTLBuffer` behind a draw argument, or `nothing` if it is not one."""
metal_buffer(x::MTLm.MTLBuffer) = x
# `pointer`, not `x.data[]`: an `MtlArray` may be a VIEW over a pool region — a
# transient's storage always is — and the buffer is the one its `MtlPtr` names.
metal_buffer(x::Metal.MtlArray) = pointer(x).buffer
metal_buffer(@nospecialize(x))  = nothing

# Three answers, and the CLEAR is asked for first.
#
# `discards(l)` is the graph's question — "does this pass need what was there?"
# — and a `Clear` answers YES just as loudly as a `Discard` does, because
# clearing is one of the two ways not to need it. Asking it first therefore
# turned every `Clear` into `DontCare`: the attachment kept whatever was in the
# arena, the geometry drew over it correctly, and only the pixels the geometry
# missed were wrong. A green-triangle test cannot see that; the clear's ALPHA
# is what gave it away.
#
# So: a clear value means Clear, and only then does `discards` distinguish
# `Discard` from `Keep`.
# One method per answer, like the rest of the state translation here, and the
# CLEAR is a type rather than a predicate — asking `clearvalue` would mean
# asking a depth op for four floats, which is an error, and asking `depthclear`
# would mean asking a colour op for one.
mtl_loadaction(::Mantle.Clear)   = MTLm.MTLLoadActionClear
mtl_loadaction(l::Mantle.LoadOp) = Mantle.discards(l) ? MTLm.MTLLoadActionDontCare :
                                                        MTLm.MTLLoadActionLoad
mtl_loadaction(::Nothing) = MTLm.MTLLoadActionDontCare

# ── The shader vocabulary, on this backend ───────────────────────────────────
#
# `@device_override`, not a plain definition: `KI.vertex_index()` has a HOST
# method that errors, and shadowing it would let a host-side call reach a GPU
# instruction on the CPU. The overlay puts these in Metal's method table, so
# they apply exactly when this compiler is running — which is the only thing
# that can decide what a builtin means.
#
# Overriding KernelInterface DIRECTLY, where the previous arrangement went
# through Mantle and a bridge in each backend. Metal.jl can reach
# KernelInterface; Lava can too; neither can reach the other's runtime. That is
# the whole reason the names moved.

# What Metal has. Generated from the list so a name added to KernelInterface and
# missed here is a `MethodError` naming it, rather than a shader that silently
# reads the wrong builtin.
const METAL_BUILTINS = (:vertex_index, :instance_index, :frag_coord_x, :frag_coord_y,
                        :frag_coord_z, :frag_coord_w, :frag_coord_xy)

for f in METAL_BUILTINS
    @eval Metal.@device_override KI.$f() = Metal.$f()
end
Metal.@device_override KI.frag_coord(dim::Integer = 1) = Metal.frag_coord(dim)

# What Metal does NOT have, and says so rather than leaving a MethodError for a
# shader compile to find:
#
#   `dFdx`/`dFdy`          Metal.jl exposes no derivative intrinsic yet; MSL
#                          has `dfdx`/`dfdy`, so this is a gap in the binding
#                          rather than in the hardware.
#   `set_point_size!`      needs `[[point_size]]` on the stage output struct,
#                          which the AIR writer does not emit yet.
#   `sample_texture_2d`    needs a bound-texture argument, which the graph's
#                          argument ABI does not carry yet.
#   `emit_vertex!`,        Metal has no geometry stage at all. Apple's
#   `end_primitive!`,      replacement is the mesh pipeline (object + mesh
#   `primitive_id_in`      stages), which Mantle does not describe.
#
# The first three are unimplemented; the last three are absent from the
# hardware. `caps` is where a caller asks which — see `supports_geometry`.

# Metal's clip space is the other one: +y is UP where the portable convention's
# (Vulkan's) is down.
#
# `@device_override`, like the builtins: a plain `KI.clip_y(y::Float32) = -y`
# here is not an override at all, it REDEFINES the declaring package's method —
# which an extension may not do, and which would also change the answer on the
# host, where nothing is being rasterised.
#
# The overlay reaches compute as well as graphics: this backend compiles both
# through the same Metal method table, so a COMPUTE shader reprojecting into a
# shadow map gets the same transform the rasteriser applied. That symmetry is
# the bug fixed in September 2026 — the vertex stage mirrored and the shadow
# lookup did not, which no test saw because a mirrored scene still looks like a
# scene and its shadow map is mirrored with it.
Metal.@device_override KI.clip_y(y::Float32) = -y

@inline flip_clip(p::Vec4f) = Vec4f(p[1], KI.clip_y(p[2]), p[3], p[4])
@inline flip_clip(p::NTuple{4,Float32}) = (p[1], KI.clip_y(p[2]), p[3], p[4])

# ── the overlay's texture table ───────────────────────────────────────────────
#
# At the END of the file because it names `MetalPassHandle`, which the render-pass
# verbs below define. A method signature is evaluated where it stands.

"""
The bound texture table on Metal.

Metal has no descriptor set: a texture and its sampler go straight onto the
encoder, in two separate slot namespaces. So this is the LIST, and
`use_bindings!` is what puts it there — which is why the two verbs are separate
in the first place, one to build a set and one to select it.
"""
struct MetalTextureBindings <: Mantle.TextureBindings
    textures::Vector{MTLm.MTLTexture}
    samplers::Vector{MTLm.MTLSamplerState}
end

Base.length(b::MetalTextureBindings) = length(b.textures)

# On THIS backend's textures, which is what `SampledTexture` carrying its
# concrete types makes possible: `bind_textures` takes no device argument, so
# without them the Vulkan method would be the only one that could exist.
function Mantle.bind_textures(
        textures::Vector{<:Mantle.SampledTexture{<:Any,<:Any,<:MetalTexture2D}})
    isempty(textures) && error("bind_textures: cannot bind an empty texture list")
    return MetalTextureBindings([st.texture.tex for st in textures],
                                [st.sampler.state for st in textures])
end

"""
    use_bindings!(emitter, compiled, bindings)

Put `bindings` on the encoder for the next draw.

Slots count from one on Mantle's side and from zero on Metal's; the setters do
the conversion, as everywhere else in this backend.
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
