# Offscreen rasterization uses native render attachments. Like compute, each
# shader reads its own argument block and addresses arrays in the device heaps.

using Mantle: BGRA, RGBA, N0f8, Depth32, SVector

Mantle.indexbuffer(d::WebGPUDevice, indices::AbstractVector{UInt32}) = Mantle.devicearray(d, indices)

Mantle.nativeformat(::WebGPUAPI, ::Type{BGRA{N0f8}}, srgb::Bool) = srgb ? "bgra8unorm-srgb" : "bgra8unorm"
Mantle.nativeformat(::WebGPUAPI, ::Type{RGBA{N0f8}}, srgb::Bool) = srgb ? "rgba8unorm-srgb" : "rgba8unorm"
Mantle.nativeformat(::WebGPUAPI, ::Type{RGBA{Float16}}, ::Bool) = "rgba16float"
Mantle.nativeformat(::WebGPUAPI, ::Type{RGBA{Float32}}, ::Bool) = "rgba32float"
Mantle.nativeformat(::WebGPUAPI, ::Type{Depth32}, ::Bool) = "depth32float"
Mantle.nativeformat(::WebGPUAPI, ::Type{Float32}, ::Bool) = "r32float"
Mantle.nativeformat(::WebGPUAPI, ::Type{Float16}, ::Bool) = "r16float"
Mantle.nativeformat(::WebGPUAPI, ::Type{UInt32}, ::Bool) = "r32uint"
Mantle.nativeformat(::WebGPUAPI, ::Type{SVector{2,UInt32}}, ::Bool) = "rg32uint"
Mantle.nativeformat(::WebGPUAPI, ::Type{NTuple{2,Float32}}, ::Bool) = "rg32float"
Mantle.nativeformat(::WebGPUAPI, ::Type{NTuple{2,Float16}}, ::Bool) = "rg16float"
Mantle.nativeformat(::WebGPUAPI, ::Type{UInt8}, ::Bool) = "r8unorm"

const FORMAT_TYPES = Dict(
    "bgra8unorm" => BGRA{N0f8}, "bgra8unorm-srgb" => BGRA{N0f8},
    "rgba8unorm" => RGBA{N0f8}, "rgba8unorm-srgb" => RGBA{N0f8},
    "rgba16float" => RGBA{Float16}, "rgba32float" => RGBA{Float32},
    "depth32float" => Float32, "r32float" => Float32, "r16float" => Float16,
    "r32uint" => UInt32, "rg32uint" => SVector{2,UInt32}, "r8unorm" => UInt8,
    "rg32float" => NTuple{2,Float32}, "rg16float" => NTuple{2,Float16})
Mantle.eltypeof(format::String) = FORMAT_TYPES[format]

mutable struct WebGPUAttachment
    dev::WebGPUDevice
    id::Int
    dims::Dims{2}
    format::String
end

function gpu_texture(d::WebGPUDevice, w::Int, h::Int, format::String, method, args...)
    id = lock(() -> (d.nbuffers += 1), d.lock)
    post(d, method, d.id, id, [w, h], format, args...)
    return finalizer(WebGPUAttachment(d, id, (w, h), format)) do a
        lock(a.dev.freedlock) do
            push!(a.dev.freedtextures, a.id)
        end
    end
end

attachment(d, w, h, format) = gpu_texture(d, w, h, format, "createAttachment")

struct WebGPUTexture2D{T} <: Mantle.Texture2D{T}
    texture::WebGPUAttachment
    levels::Int
end

Base.size(t::WebGPUTexture2D) = t.texture.dims
Mantle.mipmapped(t::WebGPUTexture2D) = t.levels > 1

Mantle.maketexture(d::WebGPUDevice, ::Type{T}, format::String, w::Int, h::Int, levels::Int) where {T} =
    WebGPUTexture2D{T}(gpu_texture(d, w, h, format, "createTexture", levels), levels)

function Mantle.copytexels!(t::WebGPUTexture2D{T}, data::AbstractMatrix{T}) where {T}
    a, d = t.texture, t.texture.dev
    bytes = copy(reinterpret(UInt8, vec(Array(data))))
    Mantle.flush!(d)
    GC.@preserve a post(d, "uploadTexture", d.id, a.id, bytes, a.dims[1] * sizeof(T))
    return t
end

struct WebGPUSampler <: Mantle.Sampler
    dev::WebGPUDevice
    descriptor::Dict{String,Any}
    filtering::Bool
end

const ADDRESS_MODES = (clamp = "clamp-to-edge", mirror = "mirror-repeat", repeat = "repeat")

function Mantle.makesampler(d::WebGPUDevice, filter::Symbol, wrap::Symbol, anisotropy::Float32)
    wrap === :clamp_border && throw(ArgumentError("WebGPU does not have clamp-to-border samplers"))
    addr = getproperty(ADDRESS_MODES, wrap)
    samples = filter === :linear ? clamp(round(Int, anisotropy), 1, 16) : 1
    desc = Dict{String,Any}("addressModeU" => addr, "addressModeV" => addr,
                            "minFilter" => String(filter), "magFilter" => String(filter),
                            "mipmapFilter" => String(filter), "maxAnisotropy" => samples)
    return WebGPUSampler(d, desc, filter === :linear)
end

struct WebGPUTextureBindings <: Mantle.TextureBindings
    textures::Vector{Mantle.SampledTexture}
end

function Mantle.bind_textures(textures::AbstractVector{<:Mantle.SampledTexture{<:Any,2,<:WebGPUTexture2D}})
    return WebGPUTextureBindings(Mantle.SampledTexture[textures...])
end

function texturetype(d::WebGPUDevice, t::Mantle.SampledTexture)
    a = t.texture.texture
    a.dev === d && t.sampler.dev === d || throw(ArgumentError("texture and sampler must belong to the render device"))
    if a.format in ("r32float", "rg32float", "rgba32float") && !("float32-filterable" in d.info["features"])
        t.sampler.filtering && throw(ArgumentError("this WebGPU adapter cannot filter Float32 textures"))
        return "unfilterable-float"
    end
    return "float"
end

texturetypes(d, ::Nothing) = ()
texturetypes(d, b::WebGPUTextureBindings) = Tuple(texturetype(d, t) for t in b.textures)

Mantle.target_extent(a::WebGPUAttachment) = a.dims
Mantle.target_format(a::WebGPUAttachment) = a.format
Mantle.target_view(a::WebGPUAttachment) = a
Mantle.target_image(a::WebGPUAttachment) = a

struct WebGPUFramebuffer <: Mantle.Framebuffer
    color::WebGPUAttachment
    depth::Union{Nothing, WebGPUAttachment}
end

function Mantle.makeframebuffer(d::WebGPUDevice, w::Int, h::Int, color::String, depth)
    return WebGPUFramebuffer(attachment(d, w, h, color),
                             depth === nothing ? nothing : attachment(d, w, h, depth))
end

Mantle.colorimage(fb::WebGPUFramebuffer) = fb.color
Mantle.depthimage(fb::WebGPUFramebuffer) = fb.depth
Mantle.depthformat(fb::WebGPUFramebuffer) = fb.depth === nothing ? nothing : fb.depth.format
Mantle.target_format(fb::WebGPUFramebuffer) = fb.color.format
Mantle.target_extent(fb::WebGPUFramebuffer) = fb.color.dims
Mantle.target_image(fb::WebGPUFramebuffer) = fb.color
Mantle.target_view(fb::WebGPUFramebuffer) = fb.color
Base.size(fb::WebGPUFramebuffer) = fb.color.dims

function Mantle.readback_framebuffer(fb::WebGPUFramebuffer)
    return readattachment(fb.color, eltype(fb))
end

function readattachment(a::WebGPUAttachment, ::Type{T}) where {T}
    d = a.dev
    commands, args = takebatch!(d)
    result = GC.@preserve a call(d, "readAttachment", d.id, a.id, a.dims[1] * sizeof(T),
                                commands, d.args, args)
    submitted!(d)
    return reshape(copy(reinterpret(T, received(result))), a.dims)
end

# WebGPU owns the physical texture storage. The planner accounts for the images
# while each transient keeps the native texture it materialized.
struct WebGPUImageArena end
const WebGPUImageReq = NamedTuple{(:size, :alignment), Tuple{Int,Int}}
const WebGPUTransientImage{T} =
    Mantle.TransientImage{T,String,Nothing,Union{Nothing,WebGPUAttachment},WebGPUImageReq}

Mantle.imageusage(::WebGPUDevice, ::Type) = nothing
Mantle.constraintof(::WebGPUDevice, ::Mantle.Images, ts) = nothing
Mantle.rawalloc(::WebGPUDevice, ::Mantle.Images, ::Int, ::Nothing) = WebGPUImageArena()
Mantle.rawfree(::WebGPUDevice, ::WebGPUImageArena) = nothing

image_requirements(::Type{T}, w, h) where {T} = (size = w * h * sizeof(T), alignment = HEAP_ALIGN)

function Mantle.makeimage(::WebGPUDevice, ::Type{T}, w::Int, h::Int, srgb::Bool, ::Nothing, source) where {T}
    fmt = Mantle.pixelformat(WebGPUAPI(), T; srgb)
    return WebGPUTransientImage{T}(w, h, fmt, nothing, nothing, image_requirements(T, w, h),
                                  typemax(Int), 0, nothing, nothing, source)
end

function Mantle.remakeimage!(::WebGPUDevice, t::WebGPUTransientImage{T}) where {T}
    t.req = image_requirements(T, t.width, t.height)
    t.image = t.view = nothing
    return t
end

function Mantle.materialize!(d::WebGPUDevice, t::WebGPUTransientImage, arena::WebGPUImageArena, offset::Int)
    if t.image === nothing
        t.image = t.view = attachment(d, t.width, t.height, t.format)
    end
    t.memory = arena
    return t
end

Mantle.readback_target(::WebGPUDevice, t::WebGPUTransientImage{T}, ::Type) where {T} =
    readattachment(t.image, T)

function Mantle.copy_target!(d::WebGPUDevice, dst::WebGPUArray, src::WebGPUTransientImage{T};
                             region = nothing) where {T}
    x, y, w, h = region === nothing ? (0, 0, src.width, src.height) : region
    row = w * sizeof(T)
    iszero(row % 4) || throw(ArgumentError("WebGPU texture readback rows require four-byte alignment"))
    offset = heapoffset(dst)
    iszero(offset % 4) || throw(ArgumentError("WebGPU texture readback destination is not four-byte aligned"))
    cmd = Dict{String,Any}("op" => "copyImage", "src" => src.image.id,
                           "dst" => heapbuffer(d, heapindex(dst)), "offset" => offset,
                           "rowBytes" => row, "region" => [x, y, w, h])
    enqueue!(d, cmd)
    return nothing
end

const BLEND_FACTORS = (one = "one", srcalpha = "src-alpha", oneminussrcalpha = "one-minus-src-alpha")
const DEPTH_COMPARISONS = (less = "less", lessequal = "less-equal", greater = "greater", always = "always")

primitive_name(::Mantle.TriangleList) = "triangle-list"
primitive_name(::Mantle.TriangleStrip) = "triangle-strip"
primitive_name(::Mantle.LineList) = "line-list"
primitive_name(::Mantle.LineStrip) = "line-strip"
primitive_name(::Mantle.PointList) = "point-list"
cull_mode(::Mantle.NoCull) = "none"
cull_mode(::Mantle.CullBack) = "back"
cull_mode(::Mantle.CullFront) = "front"

blendcomponent(src, dst) =
    Dict("srcFactor" => getproperty(BLEND_FACTORS, src), "dstFactor" => getproperty(BLEND_FACTORS, dst))

function color_target(format, mode)
    target = Dict{String,Any}("format" => format, "writeMask" => Mantle.writescolor(mode) ? 15 : 0)
    factors = Mantle.blendfactors(mode)
    if factors !== nothing
        src, dst, sa, da = factors
        target["blend"] = Dict("color" => blendcomponent(src, dst), "alpha" => blendcomponent(sa, da))
    end
    return target
end

struct WebGPUDraw
    pipeline::Int
    vertex::Lava.PushConstantInfo
    fragment::Union{Nothing, Lava.PushConstantInfo}
    vertex_args::Tuple
    fragment_args::Tuple
    texture_types::Tuple
end

function raster_shader(f, tt, stage; flip_y, heaps)
    features = Lava.TargetFeatures(webgpu = true)
    job = Lava.lava_gfx_job(f, tt; features)
    s = Lava.lava_compile_gfx_shader(f, tt; stage, validate = false, job)
    return Lava.spirv_to_wgsl(reinterpret(UInt32, s.spirv_bytes); heaps, flip_y), s.push_info
end

function raster_pipeline(d::WebGPUDevice, p::Mantle.GraphicsPipeline, colors, depth, vtt, ftt, textures;
                         flip_y::Bool, indexed::Bool)
    vf = Mantle.stagefunction(p.vertex)
    ff = Mantle.stagefunction(p.fragment)
    flats = Mantle.flatoutputs(p.vertex)
    vid = Lava.wantsvertexindex(vf, fieldtypes(vtt))
    vfn = Lava.VertexWrapper{typeof(vf), flats, vid}()
    ffn = Lava.FragmentWrapper{typeof(ff), Mantle.outputtype(p.vertex), flats}()
    key = (typeof(p), typeof(vfn), typeof(ffn), vtt, ftt, Tuple(colors), depth, flip_y, indexed, textures)
    hit = lock(() -> get(d.renderpipelines, key, nothing), d.lock)
    hit === nothing || return hit

    heaps = length(d.heaps)
    vert, vlayout = raster_shader(vfn, vtt, :vertex; flip_y, heaps)
    frag, flayout = isempty(colors) ? (nothing, nothing) :
                    raster_shader(ffn, ftt, :fragment; flip_y, heaps)
    targets = [color_target(format, Mantle.attachmentblend(p.blend, i, endswith(format, "uint")))
               for (i, format) in enumerate(colors)]
    depthstate = depth === nothing ? nothing :
        Dict("format" => depth, "depthWriteEnabled" => Mantle.depthwrites(p.depth),
             "depthCompare" => getproperty(DEPTH_COMPARISONS, something(Mantle.depthtest(p.depth), :always)))
    primitive = Dict("topology" => primitive_name(p.topology), "cullMode" => cull_mode(p.cull),
                     "frontFace" => (flip_y ? "cw" : "ccw"))
    indexed && endswith(primitive["topology"], "strip") && (primitive["stripIndexFormat"] = "uint32")
    pid = lock(() -> (d.npipelines += 1), d.lock)
    result = call(d, "createRenderPipeline", d.id, pid, vert, frag, targets, depthstate, primitive,
                  Lava.WEBGPU_FRAGMENT_ARGS_BINDING, String[textures...])
    if !result["ok"]
        msgs = join(("$(m["stage"]) line $(m["line"]):$(m["column"]) $(m["message"])"
                     for m in result["messages"]), "\n")
        error("the page rejected a WebGPU render pipeline:\n$msgs")
    end
    entry = (pid, vlayout, flayout)
    lock(() -> (d.renderpipelines[key] = entry), d.lock)
    return entry
end

shaderargs(d, args) = Mantle.argwise(a -> KI.argconvert(d, Mantle.resolve(d, a)), args)

function Mantle.compile_draw(d::WebGPUDevice, p::Mantle.GraphicsPipeline, colors, depth, vert, frag;
                             bindings = nothing, flip_y::Bool = true, indexed::Bool = false)
    textures = texturetypes(d, bindings)
    length(textures) >= Mantle.ntextures(p.fragment) ||
        throw(ArgumentError("the WebGPU fragment pipeline requires $(Mantle.ntextures(p.fragment)) sampled textures"))
    vtt, ftt = typeof(shaderargs(d, vert)), typeof(shaderargs(d, frag))
    pipeline = Base.invokelatest(raster_pipeline, d, p, colors, depth, vtt, ftt, textures; flip_y, indexed)
    return WebGPUDraw(pipeline..., vert, frag, textures)
end

mutable struct WebGPUPass
    dev::WebGPUDevice
    targets::Tuple
    depth::Union{Nothing,WebGPUAttachment}
    command::Dict{String,Any}
    dims::Dims{2}
    viewport::NTuple{4,Float64}
    scissor::NTuple{4,Int}
    bindings::Union{Nothing,WebGPUTextureBindings}
    usedtextures::Vector{WebGPUTextureBindings}
    temporaries::Vector{Any}
end

clearcolor(load) = Mantle.clearvalue(load)
clearcolor(::Mantle.DiscardOp) = (0f0, 0f0, 0f0, 0f0)
cleardepth(load) = Mantle.depthclear(load)
cleardepth(::Nothing) = nothing
cleardepth(::Mantle.DiscardOp) = 1f0

function Mantle.begin_render_pass!(d::WebGPUDevice, targets, loads, depth, depth_load)
    dims = Mantle.target_extent(isempty(targets) ? depth : first(targets))
    w, h = dims
    colors = Dict{String,Any}[Dict("id" => a.id, "clear" => clearcolor(load)) for (a, load) in zip(targets, loads)]
    dep = depth === nothing ? nothing : Dict("id" => depth.id, "clear" => cleardepth(depth_load))
    cmd = Dict{String,Any}("op" => "render", "targets" => colors, "depth" => dep,
                          "draws" => Dict{String,Any}[], "args" => UInt8[],
                          "fragBinding" => Lava.WEBGPU_FRAGMENT_ARGS_BINDING)
    return WebGPUPass(d, Tuple(targets), depth, cmd, dims, (0., 0., Float64(w), Float64(h)), (0, 0, w, h),
                      nothing, WebGPUTextureBindings[], Any[])
end

function Mantle.use_bindings!(h::WebGPUPass, d::WebGPUDraw, bindings::WebGPUTextureBindings)
    texturetypes(h.dev, bindings) == d.texture_types ||
        throw(ArgumentError("texture bindings differ from the WebGPU pipeline layout"))
    h.bindings = bindings
    push!(h.usedtextures, bindings)
    return nothing
end

function drawtextures(h::WebGPUPass, d::WebGPUDraw)
    isempty(d.texture_types) && return []
    b = h.bindings
    b === nothing && throw(ArgumentError("this WebGPU draw requires texture bindings"))
    texturetypes(h.dev, b) == d.texture_types || throw(ArgumentError("WebGPU draw uses an incompatible texture layout"))
    return [Dict("id" => t.texture.texture.id, "sampler" => t.sampler.descriptor) for t in b.textures]
end

function Mantle.setscissor!(h::WebGPUPass, x::Integer, y::Integer, w::Integer, ht::Integer)
    tw, th = h.dims
    sx, sy = clamp(x, 0, tw), clamp(y, 0, th)
    h.scissor = (sx, sy, clamp(x + w, sx, tw) - sx, clamp(y + ht, sy, th) - sy)
    return nothing
end

function Mantle.setviewport!(h::WebGPUPass, x::Real, y::Real, w::Real, ht::Real)
    h.viewport = (Float64(x), Float64(ht < 0 ? y + ht : y), Float64(w), Float64(abs(ht)))
    Mantle.setscissor!(h, Mantle.scissorrect(x, y, w, ht)...)
    return nothing
end

function pushstage!(bytes::Vector{UInt8}, layout, args)
    layout === nothing && return 0
    offset = Mantle.argalign(length(bytes))
    append!(bytes, zeros(UInt8, offset - length(bytes)))
    append!(bytes, packargs(layout, args))
    return offset
end

function Mantle.record_draw!(h::WebGPUPass, d::WebGPUDraw, args, count::Union{Integer,Mantle.Commands};
                             instances::Integer = 1, indices = nothing)
    (h.scissor[3] == 0 || h.scissor[4] == 0) && return nothing
    indirect = if count isa Mantle.Commands
        indices === nothing || throw(ArgumentError("WebGPU DrawIndirectCommand describes a non-indexed draw"))
        a = Mantle.resolve(h.dev, count.resource)
        isempty(a) && return nothing
        [heapindex(a), heapoffset(a)]
    else
        (count == 0 || instances == 0) && return nothing
        nothing
    end
    vert = shaderargs(h.dev, d.vertex_args)
    frag = d.fragment === nothing ? () : shaderargs(h.dev, d.fragment_args)
    bytes = h.command["args"]::Vector{UInt8}
    voff = pushstage!(bytes, d.vertex, vert)
    foff = pushstage!(bytes, d.fragment, frag)
    idx = if indices === nothing
        nothing
    else
        a = Mantle.resolve(h.dev, indices)
        eltype(a) === UInt32 || throw(ArgumentError("WebGPU indices must have element type UInt32"))
        count <= length(a) || throw(BoundsError(a, count))
        [heapindex(a), heapoffset(a), sizeof(a)]
    end
    push!(h.command["draws"], Dict{String,Any}("pipeline" => d.pipeline, "offsets" => [voff, foff],
          "viewport" => collect(h.viewport), "scissor" => collect(h.scissor),
          "count" => (indirect === nothing ? count : 0), "indirect" => indirect,
          "instances" => instances, "index" => idx, "textures" => drawtextures(h, d)))
    return nothing
end

function Mantle.end_render_pass!(h::WebGPUPass)
    GC.@preserve h enqueue!(h.dev, h.command)
    return nothing
end

Mantle.supports_graphics(::WebGPUDevice) = true

# Native raster stages have read-only heap bindings. The geometry fallback uses
# compute, so conservatively order any storage effects of that body.
Mantle.vertextouches(::WebGPUDevice, ::Mantle.GraphicsPipeline, args::Tuple) =
    fill(Mantle.READ, length(args))
Mantle.vertextouches(::WebGPUDevice, ::Mantle.GeometryPipeline, args::Tuple) =
    fill(Mantle.OPAQUE, length(args))
Mantle.fragmenttouches(::WebGPUDevice, ::Mantle.GraphicsPipeline, args::Tuple) =
    fill(Mantle.READ, length(args))

function Mantle.compiledraw(c::Mantle.Compile{WebGPUDevice}, p::Mantle.Pass, d, argoff::Int)
    dev = c.graph.dev
    colors = map(Mantle.target_format, p.targets)
    depth = p.depth === nothing ? nothing : Mantle.target_format(p.depth)
    Mantle.checkdepthtarget(d.shader.depth, depth !== nothing)
    Mantle.checkstages(dev, d.shader)
    vert = Mantle.argmap(a -> Mantle.resolve(dev, a), Mantle.drawargs(d.args))
    frag = Mantle.argmap(a -> Mantle.resolve(dev, a), d.frag_args)
    viewport = Mantle.drawviewport(p, d)
    flip_y = viewport === nothing || viewport[4] >= 0
    indices = d.indices === nothing ? nothing : Mantle.resolve(dev, d.indices)
    compiled = Mantle.compile_draw(dev, d.shader, colors, depth, vert, frag;
                                  d.bindings, flip_y, indexed = indices !== nothing)
    args = Mantle.packedargs(d.args, vert, frag)
    return Mantle.CompiledDraw(compiled, d.shader, args, Mantle.bakedcount(c, d.count),
                               viewport, d.scissor, d.bindings, indices, d.instances, argoff, 0)
end

function Mantle.beginrender!(e::WebGPURun, pl::Mantle.Plan, pp::Mantle.PassPlan)
    p = pp.pass
    targets = map(Mantle.target_view, p.targets)
    depth = p.depth === nothing ? nothing : Mantle.target_view(p.depth)
    return Mantle.begin_render_pass!(e.dev, targets, p.loads, depth, p.depth_load)
end

function Mantle.drawcompiled!(::WebGPURun, h::WebGPUPass, d::Mantle.CompiledDraw)
    draw = d.compiled
    args = Mantle.drawargs(d.args)
    if d.args isa Mantle.DrawBinding
        draw = rebind_draw(draw, args)
    end
    Mantle.record_draw!(h, draw, args, Mantle.drawcount(Mantle.boundcount(d.args, d.count));
                        instances = Mantle.boundinstances(d.args, d.instances),
                        indices = Mantle.boundindices(d.args, d.indices))
    return nothing
end

function rebind_draw(draw::WebGPUDraw, args)
    vert = isempty(draw.vertex_args) ? () : args
    frag = isempty(draw.fragment_args) ? () : args
    return WebGPUDraw(draw.pipeline, draw.vertex, draw.fragment, vert, frag, draw.texture_types)
end

Mantle.endrender!(::WebGPURun, h::WebGPUPass) = Mantle.end_render_pass!(h)

function Mantle.emitcopy!(e::WebGPURun, ::Mantle.Plan, pp::Mantle.PassPlan)
    p = pp.pass
    return Mantle.copy_target!(e.dev, Mantle.storage(p.dst), Mantle.first_target(p); region = p.viewport)
end

include("geometry.jl")
include("window.jl")
