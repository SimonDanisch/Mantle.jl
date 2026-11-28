# A compute workgroup writes one meshlet. A conventional vertex shader fetches
# vertices in primitive order. Unused slots clip out without a host readback.

uses_mesh_fallback(dev, shader) = false

struct SoftwareMeshOutput{V,P,F,C}
    vertices::V
    primitives::P
    flat::F
    counts::C
    group::Int32
    vertex_base::Int32
    primitive_base::Int32
    max_primitives::Int32
end

@inline mesh_output_value(a, v) = eltype(a)(NamedTuple{fieldnames(eltype(a))}(v))

@inline function Lava.set_mesh_vertex!(o::SoftwareMeshOutput, slot::Integer, v::NamedTuple)
    @inbounds o.vertices[o.vertex_base + Int32(slot)] = mesh_output_value(o.vertices, v)
    flat = Base.structdiff(v, NamedTuple{fieldnames(eltype(o.vertices))})
    if !isempty(flat)
        # Preserve the native provoking-vertex convention for shaders that
        # write flat fields with their vertices. Shared vertices use the
        # explicit primitive writer instead.
        s = Int32(slot) - Int32(1)
        arity = Int32(fieldcount(eltype(o.primitives)))
        if s % arity == 0
            p = s ÷ arity + Int32(1)
            p <= o.max_primitives && Lava.set_mesh_primitive_data!(o, p, flat)
        end
    end
    return nothing
end

@generated function Lava.set_mesh_primitive_data!(o::SoftwareMeshOutput, slot::Integer,
                                                v::NamedTuple{Names}) where {Names}
    writes = [:(@inbounds getfield(o.flat, $(QuoteNode(name)))[i] = getfield(v, $(QuoteNode(name))))
              for name in Names]
    return quote
        Base.@_inline_meta
        i = o.primitive_base + Int32(slot)
        $(writes...)
        return nothing
    end
end

@inline function Lava.set_mesh_triangle!(o::SoftwareMeshOutput, slot::Integer, a::Integer, b::Integer, c::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(slot)] = (Int32(a), Int32(b), Int32(c))
    return nothing
end

@inline function Lava.set_mesh_line!(o::SoftwareMeshOutput, slot::Integer, a::Integer, b::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(slot)] = (Int32(a), Int32(b))
    return nothing
end

@inline function Lava.set_mesh_point!(o::SoftwareMeshOutput, slot::Integer, a::Integer)
    @inbounds o.primitives[o.primitive_base + Int32(slot)] = (Int32(a),)
    return nothing
end

@inline function Lava.set_mesh_outputs!(o::SoftwareMeshOutput, nv::Integer, np::Integer)
    @inbounds o.counts[o.group] = (Int32(nv), Int32(np))
    return nothing
end

function expand_mesh!(program, vertices, primitives, flat, counts, maxv::Int32, maxp::Int32,
                      args::Vararg{Any,N}) where {N}
    group = Int32(KI.get_group_id().x)
    base = group - Int32(1)
    out = SoftwareMeshOutput(vertices, primitives, flat, counts, group, base * maxv, base * maxp, maxp)
    splatcall(program, (out,), args)
    return nothing
end

@inline mesh_flat_values(flat, i) = map(a -> (@inbounds a[i]), flat)

function expanded_mesh_vertex(vid::VertexIndex, vertices, primitives, flat, counts,
                              maxv::Int32, maxp::Int32, outside, fragargs...)
    arity = Int32(fieldcount(eltype(primitives)))
    p, lane = divrem(vid.value - Int32(1), arity)
    group, slot = divrem(p, maxp)
    @inbounds nv, np = counts[group + Int32(1)]
    slot < np || return outside
    primitive_slot = p + Int32(1)
    @inbounds i = primitives[primitive_slot][lane + Int32(1)]
    Int32(1) <= i <= nv || return outside
    @inbounds v = vertices[group * maxv + i]
    values = merge(v, mesh_flat_values(flat, primitive_slot))
    return typeof(outside)(NamedTuple{fieldnames(typeof(outside))}(values))
end

struct ExpandedMeshFragment{F}
    body::F
end
Base.nameof(f::ExpandedMeshFragment) = Symbol(nameof(f.body), :_expanded_mesh)
@inline function (f::ExpandedMeshFragment)(inputs, vertices, primitives, flat, counts,
                                           maxv, maxp, outside, args::Vararg{Any,N}) where {N}
    return splatcall(f.body, (inputs,), args)
end

mesh_cull_value(::Type{T}) where {T} = zero(T)
mesh_cull_value(::Type{T}) where {T<:Tuple} = map(mesh_cull_value, fieldtypes(T))

mesh_list_topology(::Union{TriangleList,TriangleStrip}) = TriangleList()
mesh_list_topology(::Union{LineList,LineStrip}) = LineList()
mesh_list_topology(::PointList) = PointList()

function software_mesh_stages(allocate, dev, p::MeshPipeline, groups::Int, fragargs)
    maxgroups = first(KI.max_num_groups(dev))
    groups <= maxgroups || throw(ArgumentError(
        "software mesh drawing supports at most $maxgroups groups per draw, got $groups"))
    m = p.mesh
    cfg = meshconfig(p)
    maxv, maxp = Int32(cfg.max_vertices), Int32(cfg.max_primitives)
    topology = mesh_list_topology(cfg.topology)
    arity = primitivevertices(topology)
    slots = Base.checked_mul(groups, cfg.max_primitives)
    nvertices = Base.checked_mul(groups, cfg.max_vertices)
    count = Base.checked_mul(slots, arity)
    count <= typemax(Int32) && nvertices <= typemax(Int32) ||
        throw(ArgumentError("software mesh output exceeds 32-bit indexing"))

    V = outputtype(m)
    flat = flatoutputs(m)
    smooth = (:position, smoothoutputs(m)...)
    VT = NamedTuple{smooth,Tuple{map(k -> fieldtype(V, k), smooth)...}}
    buffers = (allocate(VT, nvertices), allocate(NTuple{arity,Int32}, slots),
               NamedTuple{flat}(map(k -> allocate(fieldtype(V, k), slots), flat)),
               allocate(NTuple{2,Int32}, groups))

    outside = V((Vec4f(0f0, 0f0, 2f0, 1f0), map(mesh_cull_value, Base.tail(fieldtypes(V)))...))
    drawargs = (buffers..., maxv, maxp, outside, fragargs...)
    vertex = VertexShader(expanded_mesh_vertex; outputs = stageoutputs(m))
    fragment = isempty(fragargs) ? p.fragment :
        FragmentShader(ExpandedMeshFragment(stagefunction(p.fragment)); textures = ntextures(p.fragment))
    pipeline = GraphicsPipeline(; vertex, fragment, p.blend, p.cull, p.depth, topology)
    return (; pipeline, drawargs, fragargs = isempty(fragargs) ? () : drawargs, count,
            compute = (stagefunction(m), buffers..., maxv, maxp),
            lanes = Base.checked_mul(groups, cfg.threads), threads = cfg.threads)
end

function mesh_dispatch(dev, p::MeshPipeline, args, groups, indices, instances)
    groups isa Integer || throw(ArgumentError("software mesh drawing requires an integer group count"))
    groups >= 0 || throw(ArgumentError("mesh group count must be nonnegative"))
    instances == 1 || throw(ArgumentError("software mesh drawing takes one instance per draw"))
    indices === nothing || throw(ArgumentError("a mesh stage emits its own indices"))
    return p, args, Int(groups)
end

function mesh_dispatch(dev, p::GeometryPipeline, args, count, indices, instances)
    count isa Integer || throw(ArgumentError("software geometry drawing requires an integer count"))
    count >= 0 || throw(ArgumentError("vertex count must be nonnegative"))
    instances == 1 || error("a lowered geometry pipeline draws one instance per draw, got $instances")
    requirevertexindex(stagefunction(p.vertex), devicetype(dev, args))
    mesh = lower_geometry_to_mesh(p; indexed = indices !== nothing)
    indices === nothing || count <= length(indices) || throw(BoundsError(indices, count))
    meshargs = indices === nothing ? args : (args..., indices)
    return mesh, meshargs, primitivecount(p.topology, Int(count))
end

function draw_software_mesh!(h::PassHandle, shader, args, count;
                             frag_args, viewport, scissor, bindings, indices, instances)
    args isa DrawBinding && throw(ArgumentError("software mesh drawing requires fixed draw arguments"))
    g = h.graph
    mesh, meshargs, groups = mesh_dispatch(g.dev, shader, args, count, indices, instances)
    checkstages(g.dev, mesh)
    groups == 0 && return nothing
    draw = software_mesh_stages((T, n) -> Transient.Buffer(g, T, n), g.dev, mesh, groups, frag_args)
    dispatch!(g, expand_mesh!, (draw.compute..., meshargs...), draw.lanes;
              group = draw.threads, name = h.pass.name * " mesh expansion")
    # The draw declares reads of the generated buffers. Install their writer
    # before the render pass so the graph derives compute-to-graphics hazards.
    renderindex = findfirst(p -> p === h.pass, g.passes)
    @assert renderindex !== nothing
    insert!(g.passes, renderindex, pop!(g.passes))
    return draw!(h, draw.pipeline, draw.drawargs, draw.count;
                 frag_args = draw.fragargs, viewport, scissor, bindings)
end

function draw_software_mesh!(dev::Device, shader, target, count;
                             args, frag_args, instances, clear_color, depth_clear, indices, bindings)
    mesh, meshargs, groups = mesh_dispatch(dev, shader, args, count, indices, instances)
    checkstages(dev, mesh)
    checkdepthtarget(mesh.depth, depthformat(target) !== nothing)
    groups == 0 && return pass!(_ -> nothing, dev, target; clear = clear_color, depth_clear)
    draw = software_mesh_stages((T, n) -> devicearray(dev, T, n), dev, mesh, groups, frag_args)
    resolved = argmap(a -> resolve(dev, a), meshargs)
    launch = ArrayLaunch(expand_mesh!, (draw.compute..., resolved...), draw.lanes, draw.threads)
    runlaunches!(dev, (launch,))
    return draw!(dev, draw.pipeline, target, draw.count;
                 args = draw.drawargs, frag_args = draw.fragargs, clear_color, depth_clear, bindings)
end
