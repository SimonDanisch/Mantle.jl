# Geometry bodies emit through Mantle's mesh lowering. A compute pass writes
# their output to heap arrays, and a regular vertex stage pulls those vertices
# in primitive order. Unused output slots are clipped by the vertex stage.

Mantle.supports_geometry_pipeline(::WebGPUDevice) = true
Mantle.supports_mesh_pipeline(::WebGPUDevice) = true
Mantle.uses_mesh_fallback(::WebGPUDevice, ::Mantle.MeshPipeline) = true

struct GeometryOutput{V,P,F,C}
    vertices::V
    primitives::P
    flat::F
    counts::C
    group::Int32
    vertex_base::Int32
    primitive_base::Int32
end

@inline outputconvert(a, v) = eltype(a)(NamedTuple{fieldnames(eltype(a))}(v))

@inline function Lava.set_mesh_vertex!(o::GeometryOutput, i::Integer, v::NamedTuple)
    @inbounds o.vertices[o.vertex_base + Int32(i)] = outputconvert(o.vertices, v)
    return nothing
end

@inline function Lava.set_mesh_primitive_data!(o::GeometryOutput, i::Integer, v::NamedTuple)
    @inbounds o.flat[o.primitive_base + Int32(i)] = outputconvert(o.flat, v)
    return nothing
end

@inline function Lava.set_mesh_triangle!(o::GeometryOutput, i::Integer, a::Integer, b::Integer, c::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(i)] = (Int32(a), Int32(b), Int32(c))
    return nothing
end

@inline function Lava.set_mesh_line!(o::GeometryOutput, i::Integer, a::Integer, b::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(i)] = (Int32(a), Int32(b))
    return nothing
end

@inline function Lava.set_mesh_point!(o::GeometryOutput, i::Integer, a::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(i)] = (Int32(a),)
    return nothing
end

@inline function Lava.set_mesh_outputs!(o::GeometryOutput, ::Integer, n::Integer)
    @inbounds o.counts[o.group] = Int32(n)
    return nothing
end

function expand_geometry!(program, buffers, args, maxv::Int32, maxp::Int32, groups::Int32)
    group = Int32(Lava.lava_global_invocation_id_x()) + Int32(1)
    if group <= groups
        b = group - Int32(1)
        out = GeometryOutput(buffers..., group, b * maxv, b * maxp)
        Mantle.splatcall(Mantle.runprimitive, (program, out, group), args)
    end
    return nothing
end

function expanded_vertex(vid::Mantle.VertexIndex, vertices, primitives, flat, counts,
                         maxv::Int32, maxp::Int32, outside)
    arity = Int32(fieldcount(eltype(primitives)))
    p, lane = divrem(vid.value - Int32(1), arity)
    group, slot = divrem(p, maxp)
    @inbounds slot < counts[group + Int32(1)] || return outside
    @inbounds begin
        i = primitives[p + Int32(1)][lane + Int32(1)]
        v = vertices[group * maxv + i]
        values = merge(v, flat[p + Int32(1)])
        return typeof(outside)(NamedTuple{fieldnames(typeof(outside))}(values))
    end
end

cullvalue(::Type{T}) where {T} = zero(T)
cullvalue(::Type{T}) where {T<:Tuple} = map(cullvalue, fieldtypes(T))

mutable struct GeometryScratch
    groups::Int
    buffers::Tuple
end

function geometry_buffers!(s::GeometryScratch, d, groups, maxv, maxp,
                           ::Type{VT}, ::Type{PT}, ::Type{FT}) where {VT,PT,FT}
    if groups > s.groups
        s.buffers = (WebGPUArray{VT}(d, (groups * maxv,)),
                     WebGPUArray{PT}(d, (groups * maxp,)),
                     WebGPUArray{FT}(d, (groups * maxp,)),
                     WebGPUArray{Int32}(d, (groups,)))
        s.groups = groups
    end
    return s.buffers
end

struct WebGPUGeometryDraw
    pipeline::Mantle.GeometryPipeline
    colors::Tuple
    depth::Union{String,Nothing}
    vertex_args::Tuple
    fragment_args::Tuple
    flip_y::Bool
    bindings::Union{WebGPUTextureBindings,Nothing}
    scratch::GeometryScratch
end

function Mantle.compile_draw(d::WebGPUDevice, p::Mantle.GeometryPipeline, colors, depth, vert, frag;
                             bindings = nothing, flip_y::Bool = true, indexed::Bool = false)
    return WebGPUGeometryDraw(p, Tuple(colors), depth, vert, frag, flip_y, bindings,
                              GeometryScratch(0, ()))
end

function Mantle.use_bindings!(h::WebGPUPass, ::WebGPUGeometryDraw, bindings::WebGPUTextureBindings)
    h.bindings = bindings
    push!(h.usedtextures, bindings)
    return nothing
end

function rebind_draw(draw::WebGPUGeometryDraw, args)
    vert = isempty(draw.vertex_args) ? () : args
    frag = isempty(draw.fragment_args) ? () : args
    return WebGPUGeometryDraw(draw.pipeline, draw.colors, draw.depth, vert, frag,
                              draw.flip_y, draw.bindings, draw.scratch)
end

listtopology(::Union{Mantle.TriangleList,Mantle.TriangleStrip}) = Mantle.TriangleList()
listtopology(::Union{Mantle.LineList,Mantle.LineStrip}) = Mantle.LineList()
listtopology(::Mantle.PointList) = Mantle.PointList()

function Mantle.record_draw!(h::WebGPUPass, draw::WebGPUGeometryDraw, args, count::Integer;
                             instances::Integer = 1, indices = nothing)
    p, d = draw.pipeline, h.dev
    groups = Mantle.primitivecount(p.topology, count)
    (groups == 0 || instances == 0 || h.scissor[3] == 0 || h.scissor[4] == 0) && return nothing
    instances == 1 || throw(ArgumentError("WebGPU geometry lowering takes one instance per draw"))
    lowered = Mantle.lower_geometry_to_mesh(p; indexed = indices !== nothing)
    m = lowered.mesh
    Mantle.requirevertexindex(Mantle.stagefunction(p.vertex), typeof(shaderargs(d, draw.vertex_args)))
    cfg = Mantle.stageconfig(m)
    maxv, maxp = Int32(cfg.max_vertices), Int32(cfg.max_primitives)
    arity = Mantle.primitivevertices(cfg.topology)
    V = Mantle.outputtype(m)
    flat = Mantle.flatoutputs(m)
    smooth = (:position, Mantle.smoothoutputs(m)...)
    VT = NamedTuple{smooth,Tuple{map(k -> fieldtype(V, k), smooth)...}}
    FT = NamedTuple{flat,Tuple{map(k -> fieldtype(V, k), flat)...}}
    buffers = geometry_buffers!(draw.scratch, d, groups, maxv, maxp, VT, NTuple{arity,Int32}, FT)

    v_args = draw.vertex_args
    if indices !== nothing
        idx = Mantle.resolve(d, indices)
        count <= length(idx) || throw(BoundsError(idx, count))
        v_args = (v_args..., idx)
    end
    outside = V((Mantle.Vec4f(0f0, 0f0, 2f0, 1f0), map(cullvalue, Base.tail(fieldtypes(V)))...))
    packed = (buffers..., maxv, maxp, outside)
    vertex = Mantle.VertexShader(expanded_vertex; outputs = Mantle.stageoutputs(m))
    pipe = Mantle.GraphicsPipeline(; vertex, fragment = p.fragment, blend = p.blend,
                                    cull = p.cull, depth = p.depth, topology = listtopology(cfg.topology))
    bindings = h.bindings === nothing ? draw.bindings : h.bindings
    compiled = Mantle.compile_draw(d, pipe, draw.colors, draw.depth, packed, draw.fragment_args;
                                   bindings, draw.flip_y)
    push!(h.temporaries, buffers)
    kernel = KI.Kernel(d, expand_geometry!)
    KI.launch(kernel, (cld(groups, 64), 1, 1), (64, 1, 1),
              (Mantle.stagefunction(m), buffers, v_args, maxv, maxp, Int32(groups)))
    return Mantle.record_draw!(h, compiled, packed, groups * maxp * arity)
end
