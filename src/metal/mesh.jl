# KernelInterface's mesh verbs lowered to Metal.jl's `air.set_*_mesh` intrinsics.
# KernelInterface slots and indices are 1-based, AIR's 0-based; the conversion
# happens only here. A vertex is a NamedTuple with `position` first; the other
# fields go to numbered slots in the mesh stage's output declaration order, which
# must match `air_mesh_vertex_info` or the fragment stage reads the wrong field.
# `@device_override`, not plain methods, so the host methods over
# `HostMeshOutput` stay intact.

"""
    mesh_field_index(V, name) -> Int32

Zero-based slot of `name` in AIR's per-vertex or per-primitive plane. `position`
is skipped: it is `air.position`, which has no index.
"""
@inline function mesh_field_index(@nospecialize(V::Type), name::Symbol)
    names = fieldnames(V)
    i = findfirst(==(name), names)
    i === nothing && error("`$name` is not a field of $V")
    return Int32(i - 1 - (first(names) === :position ? 1 : 0))
end

# Splits the vertex by name at compile time, so the per-vertex write has no lookup.
@generated function write_mesh_vertex!(out::Metal.MeshPtr{V,P,NV,NP,T},
                                        slot::Int32, v::NamedTuple) where {V,P,NV,NP,T}
    calls = Expr[]
    hasfield(v, :position) ||
        error("a mesh vertex needs a `position`; got $(fieldnames(v))")
    # `flip_clip` as in `MetalVertexStage`: Mantle's clip space is Vulkan's
    # (+y down), Metal's is mirrored.
    push!(calls, :(Metal.set_position_mesh(out, slot,
                       flip_clip(mesh_vec4(Base.getfield(v, :position))))))
    # Portable shaders write `Flat` outputs per vertex, as on Vulkan, but AIR
    # has no flat mesh vertex data: flat varyings are per-primitive data
    # (`air.mesh_primitive_data`), so they are routed to the `P` plane here.
    flat = Symbol[]
    for n in fieldnames(v)
        n === :position && continue
        if hasfield(V, n)
            # Field before slot for the data intrinsics, unlike `set_position_mesh`.
            push!(calls, :(Metal.set_vertex_data_mesh(
                out, $(mesh_field_index(V, n)), slot,
                mesh_component(Base.getfield(v, $(QuoteNode(n)))))))
        elseif hasfield(P, n)
            push!(flat, n)
        else
            error("`$n` is not an output of this mesh stage: it declares " *
                  "$(Base.tail(fieldnames(V))) smooth and $(fieldnames(P)) flat")
        end
    end
    if !isempty(flat)
        # The provoking (first) vertex of primitive `slot ÷ vpp` writes its flat
        # data, as on Vulkan. Exact only when each primitive owns its vertices;
        # stages sharing vertices must use `set_mesh_primitive_data!`.
        vpp = mesh_verts_per_primitive(T)
        writes = [:(Metal.set_primitive_data_mesh(
                        out, $(mesh_field_index(P, n)), prim,
                        mesh_component(Base.getfield(v, $(QuoteNode(n))))))
                  for n in flat]
        push!(calls, quote
            if slot % $vpp == Int32(0)
                prim = slot ÷ $vpp
                $(writes...)
            end
        end)
    end
    return Expr(:block, Expr(:meta, :inline), calls..., :(return nothing))
end

@generated function write_mesh_primitive!(out::Metal.MeshPtr{V,P,NV,NP,T},
                                           slot::Int32, d::NamedTuple) where {V,P,NV,NP,T}
    calls = Expr[:(Metal.set_primitive_data_mesh(
                       out, $(mesh_field_index(P, n)), slot,
                       mesh_component(Base.getfield(d, $(QuoteNode(n))))))
                 for n in fieldnames(d)]
    return Expr(:block, Expr(:meta, :inline), calls..., :(return nothing))
end

# AIR takes plain `Float32` tuples; `Vec4f` and other spellings are converted.
@inline mesh_vec4(p::NTuple{4,Float32}) = p
@inline mesh_vec4(p) = (Float32(p[1]), Float32(p[2]), Float32(p[3]), Float32(p[4]))
@inline mesh_component(x::Union{Float32,NTuple{2,Float32},NTuple{3,Float32},
                                 NTuple{4,Float32}}) = x
@inline mesh_component(x::Real) = Float32(x)
# `ntuple(..., Val(N))`, not a generator: a generator's iteration error path needs
# `kernel_state`, which a mesh stage lacks (`unsupported call to
# julia.gpu.state_getter` at shader compile).
@inline mesh_component(x) = ntuple(i -> Float32(@inbounds x[i]), Val(length(x)))

# No `@inline` before `@device_override`: it registers an ordinary method instead
# of an overlay, which fails at shader compile with a `MethodError`.
Metal.@device_override KI.set_mesh_vertex!(out::Metal.MeshPtr, slot::Integer,
                                           v::NamedTuple) =
    write_mesh_vertex!(out, Int32(slot) - Int32(1), v)

Metal.@device_override KI.set_mesh_primitive_data!(out::Metal.MeshPtr, slot::Integer,
                                                   d::NamedTuple) =
    write_mesh_primitive!(out, Int32(slot) - Int32(1), d)

# AIR has one flat index list: zero-based primitive `p` of arity `k` starts at `k*p`.
# Indices are `UInt8`; `MeshConfig` enforces `MAX_MESH_VERTICES = 256`, so the
# truncation here is safe.

@inline function write_mesh_indices!(out::Metal.MeshPtr, base::Int32, idx::Vararg{Int32,N}) where {N}
    ntuple(Val(N)) do k
        Metal.set_index_mesh(out, base + Int32(k) - Int32(1),
                             unsafe_trunc(UInt8, idx[k] - Int32(1)))
    end
    return nothing
end

Metal.@device_override KI.set_mesh_triangle!(out::Metal.MeshPtr, slot::Integer,
                                             i0::Integer, i1::Integer, i2::Integer) =
    write_mesh_indices!(out, (Int32(slot) - Int32(1)) * Int32(3),
                         Int32(i0), Int32(i1), Int32(i2))

Metal.@device_override KI.set_mesh_line!(out::Metal.MeshPtr, slot::Integer,
                                         i0::Integer, i1::Integer) =
    write_mesh_indices!(out, (Int32(slot) - Int32(1)) * Int32(2), Int32(i0), Int32(i1))

Metal.@device_override KI.set_mesh_point!(out::Metal.MeshPtr, slot::Integer,
                                          i0::Integer) =
    write_mesh_indices!(out, Int32(slot) - Int32(1), Int32(i0))

# Metal takes only the primitive count; the vertex bound is in the object type.
# A threadgroup declaring zero primitives is discarded with its side effects.
Metal.@device_override KI.set_mesh_outputs!(out::Metal.MeshPtr, ::Integer,
                                            nprimitives::Integer) =
    Metal.set_primitive_count_mesh(out, Int32(nprimitives))

# Metal.jl's thread intrinsics are already 1-based.
Metal.@device_override KI.mesh_thread_index() =
    Int32(Metal.thread_index_in_threadgroup())
Metal.@device_override KI.mesh_group_index() =
    Int32(Metal.threadgroup_position_in_grid().x)

# `set_mesh_groups!` is absent on purpose: Metal.jl emits no `air.object` stage,
# and a `MethodError` beats a silent no-op.

"""
    mesh_object_of(p::MeshPipeline) -> Type{<:MeshObject}

The output object type the mesh stage of `p` writes through, derived from the
declaration only (smooth/flat outputs, `MeshConfig` bounds, topology), since the
fragment stage links against it before either stage compiles.
"""
function mesh_object_of(p::Mantle.MeshPipeline)
    cfg = Mantle.meshconfig(p)
    outs = Mantle.stageoutputs(p.mesh)
    vals = Mantle.valuetypes(outs)
    smooth = Mantle.smoothoutputs(p.mesh)
    flat   = Mantle.flatoutputs(p.mesh)
    # `position` is always first and never declared, as in `outputtype`.
    V = NamedTuple{(:position, smooth...),
                   Tuple{NTuple{4,Float32}, (vals[n] for n in smooth)...}}
    P = NamedTuple{flat, Tuple{(vals[n] for n in flat)...}}
    return Metal.MeshObject{V, P, cfg.max_vertices, cfg.max_primitives,
                            mesh_topology_symbol(cfg.topology)}
end

"""
    mesh_topology_symbol(topology) -> Symbol

AIR's primitive kind for a topology. Strips map to their list kind: a mesh
stage emits explicitly indexed primitives.
"""
mesh_topology_symbol(::Union{Mantle.TriangleList,Mantle.TriangleStrip}) = :triangle
mesh_topology_symbol(::Union{Mantle.LineList,Mantle.LineStrip})         = :line
mesh_topology_symbol(::Mantle.PointList)                                = :point

"""
    mesh_verts_per_primitive(kind::Symbol) -> Int32

Vertices per primitive of AIR's `kind`. Takes the symbol because
`write_mesh_vertex!` only has the `MeshObject`'s type parameter.
"""
mesh_verts_per_primitive(kind::Symbol) =
    kind === :triangle ? Int32(3) :
    kind === :line     ? Int32(2) :
    kind === :point    ? Int32(1) :
    error("unknown mesh primitive kind `$kind`")

"""
    mesh_stage_lead(p::MeshPipeline) -> Tuple

The arguments the mesh body receives before the caller's: its output object.
`compile_pipeline` and `vertextouches` must use the same signature.
"""
mesh_stage_lead(p::Mantle.MeshPipeline) = (Core.LLVMPtr{mesh_object_of(p), 7},)

"""One compiled mesh pipeline: the state, its depth state, and what it needs at draw."""
struct MetalCompiledMeshPipeline
    state::MTLm.MTLRenderPipelineState
    depth_state::Union{Nothing,MTLm.MTLDepthStencilState}
    cull::MTLm.MTLCullMode
    threads::Int
    libs::Vector{Any}
end


"""The mesh stage body, handed its arguments with `PackedArgs` splatted back out."""
struct MetalMeshStage{F} end

@generated function (::MetalMeshStage{F})(args::Vararg{Any,N}) where {F,N}
    call = Expr(:call, :(F.instance), unpacked(args, 1:N)...)
    quote
        Base.@_inline_meta
        $call
        return nothing
    end
end

"""
    compile_pipeline(device, p::MeshPipeline, color_formats, depth_format, mesh_bufs, frag_bufs)

Compile a portable [`MeshPipeline`](@ref) into a Metal render pipeline state.

`mesh_bufs`/`frag_bufs` are the Tuple types of each stage's buffer arguments; the
mesh output object is prepended here. Cached on everything baked in.
"""
function compile_pipeline(d::MetalDevice, p::Mantle.MeshPipeline,
                          color_formats::Vector{MTLm.MTLPixelFormat},
                          depth_format::Union{Nothing,MTLm.MTLPixelFormat},
                          mesh_bufs::Type, frag_bufs::Type)
    p.object === nothing ||
        error("this backend has no object stage yet: Metal.jl emits no `air.object` " *
              "program, so a MeshPipeline's mesh threadgroups are dispatched by the " *
              "host. Build the pipeline without one and pass the group count to `draw!`.")

    dev = d.dev
    cfg = Mantle.meshconfig(p)
    Obj = mesh_object_of(p)

    # The wrapper only splats `PackedArgs`.
    mesh_tt = Tuple{mesh_stage_lead(p)..., mesh_bufs.parameters...}
    mfn = MetalMeshStage{typeof(Mantle.stagefunction(p.mesh))}()
    mstage = compile_stage_function(dev, mfn, mesh_tt, :mesh,
                                    stagename(Mantle.stagefunction(p.mesh), "_ms"))

    isempty(color_formats) &&
        error("a mesh pipeline needs a colour attachment: Metal has no " *
              "depth-only mesh pipeline, and a nil fragment function is refused " *
              "for one.")
    VIn  = varying_type(p)
    FOut = fragment_output_type(color_formats)
    ntex = Mantle.ntextures(p.fragment)
    ffn = MetalFragmentStage{typeof(Mantle.stagefunction(p.fragment)), VIn, FOut, ntex}()
    frag_tt = Tuple{frag_bufs.parameters..., varying_markers(VIn)...,
                    texture_markers(ntex)..., StageOut{FOut}}
    fstage = compile_stage_function(dev, ffn, frag_tt, :fragment,
                                    stagename(Mantle.stagefunction(p.fragment), "_fs"))

    # Keyed on stage function identity: a recompiled stage is a new key.
    key = (:mesh, CacheIdentity((mstage, fstage)), Tuple(color_formats),
           depth_format, pipeline_state_key(p), cfg.threads)
    return get!(d.pipelines, key) do
        mfun, mlib = mstage
        ffun, flib = fstage

        desc = MTLm.MTLMeshRenderPipelineDescriptor()
        desc.meshFunction = mfun
        desc.fragmentFunction = ffun
        apply_attachments!(desc, color_formats, depth_format, p.blend)
        # From `MeshConfig`: Metal refuses a mismatch with the shader's width.
        desc.maxTotalThreadsPerMeshThreadgroup = cfg.threads
        state = MTLm.MTLRenderPipelineState(dev, desc)

        dstate = depthstate(dev, p.depth, depth_format)

        compiled = MetalCompiledMeshPipeline(state, dstate, mtl_cull(p.cull), cfg.threads,
                                             Any[mlib, flib])
        return compiled
    end::MetalCompiledMeshPipeline
end

# `draw!` is core's (`graphics/record.jl`), over `compile_draw` and `record_draw!`.


# Answered for the backend and the device, since a caller may hold either.
Mantle.supports_mesh_pipeline(::Metal.MetalBackend) = true
Mantle.supports_mesh_pipeline(::MetalDevice) = true

# Geometry pipelines run as mesh pipelines via `Mantle.lower_geometry_to_mesh`,
# so `supports_geometry_stage` is `false` and the same `GraphicsPipeline` still draws.

"""One draw of a lowered geometry pipeline: the description, and the baked arguments."""
mutable struct MetalCompiledGeometryDraw
    const device::MetalDevice
    const pipeline::Mantle.GraphicsPipeline
    const color_formats::Vector{MTLm.MTLPixelFormat}
    const depth_format::Union{Nothing,MTLm.MTLPixelFormat}
    const mesh::StageArgs
    const frag::StageArgs
    # The lowered mesh pipeline per indexedness, cached: lowering is pure and
    # otherwise allocates on every draw of every frame.
    lowered_plain::Any
    lowered_indexed::Any
end

MetalCompiledGeometryDraw(dev, p, cf, df, mesh, frag) =
    MetalCompiledGeometryDraw(dev, p, cf, df, mesh, frag, nothing, nothing)

"""The lowered mesh pipeline for this draw, lowered once."""
function loweredmesh(d::MetalCompiledGeometryDraw, indexed::Bool)
    if indexed
        d.lowered_indexed === nothing &&
            (d.lowered_indexed = Mantle.lower_geometry_to_mesh(d.pipeline; indexed = true))
        return d.lowered_indexed
    end
    d.lowered_plain === nothing &&
        (d.lowered_plain = Mantle.lower_geometry_to_mesh(d.pipeline; indexed = false))
    return d.lowered_plain
end

"""`groups` mesh threadgroups of `threads` threads each."""
drawmesh!(enc, groups, threads) = MTLm.draw_mesh_threadgroups!(enc, MTLm.MTLSize(groups, 1, 1),
    MTLm.MTLSize(1, 1, 1), MTLm.MTLSize(threads, 1, 1))

"""What `compile_draw` returns for a `MeshPipeline` and `record_draw!` binds: the
pipeline and both stages' baked arguments, which outlive the pass. The draw count
is a threadgroup count.
"""
struct MetalCompiledMeshDraw
    pipeline::MetalCompiledMeshPipeline
    mesh::StageArgs
    frag::StageArgs
end

function Mantle.compile_draw(d::MetalDevice, p::Mantle.MeshPipeline,
                             color_formats, depth_format, mesh_args, frag_args;
                             bindings = nothing)
    mesh, frag = StageArgs(mesh_args), StageArgs(frag_args)
    cfmts, dfmt = mtlformats(color_formats, depth_format)
    pipeline = compile_pipeline(d, p, cfmts, dfmt, buffer_types(mesh), buffer_types(frag))
    return MetalCompiledMeshDraw(pipeline, mesh, frag)
end

"""
    record_draw!(h, d::MetalCompiledMeshDraw, args, groups; instances, indices)

Record one mesh draw of `groups` threadgroups at the pipeline's width. Throws
for `indices` (the mesh stage writes its own) and for `instances != 1`.
"""
function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledMeshDraw, args, groups;
                             instances::Integer = 1, indices = nothing)
    indices === nothing || throw(ArgumentError(
        "record_draw!: a mesh pipeline takes no index buffer — the mesh stage writes " *
        "its own primitive indices. `count` is its threadgroup count."))
    instances == 1 || throw(ArgumentError(
        "record_draw!: this backend dispatches mesh threadgroups directly, so there " *
        "is no instance count to multiply them by; got instances = $(instances)."))
    rebakestages!(args, d.mesh, d.frag)
    c = d.pipeline
    setdrawstate!(h.encoder, c)
    bind_stage!(h.encoder, d.mesh, MTLm.set_mesh_bytes!, MTLm.MTLRenderStageMesh)
    bind_stage!(h.encoder, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)
    drawmesh!(h.encoder, groups, c.threads)
    return nothing
end

"""
    compile_draw(d::MetalDevice, p::GeometryPipeline, …)

Bake a geometry pipeline's arguments. The pipeline compiles at
[`record_draw!`](@ref): the lowered mesh stage takes the index buffer as an
argument, and whether a draw is indexed is known only there. The vertex body's
`VertexIndex` parameter is checked here, so a shader mistake is reported early.
"""
function Mantle.compile_draw(d::MetalDevice,
                             p::Mantle.GeometryPipeline,
                             color_formats, depth_format, vert_args, frag_args;
                             bindings = nothing)
    mesh, frag = StageArgs(vert_args), StageArgs(frag_args)
    Mantle.requirevertexindex(Mantle.stagefunction(p.vertex), bodytypes(buffer_types(mesh)))
    return MetalCompiledGeometryDraw(d, p, mtlformats(color_formats, depth_format)..., mesh, frag)
end

"""
    record_draw!(h, d::MetalCompiledGeometryDraw, args, count; instances, indices)

Record a lowered geometry draw: one mesh threadgroup per input primitive.
`count` is the vertex or index count, as for the geometry pipeline;
`Mantle.primitivecount` converts it. Throws for `instances != 1`.
"""
function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledGeometryDraw, args,
                             count::Integer; instances::Integer = 1, indices = nothing)
    instances == 1 || error(
        "a lowered geometry pipeline draws one instance, and this draw asks for " *
        "$instances. A mesh dispatch has a three-dimensional threadgroup grid and " *
        "the instance would go on its second axis, which needs a portable name for " *
        "`instance_index()` inside a mesh stage; there is none yet and nothing in " *
        "tree instances a geometry pipeline.")

    # Bind this frame's arguments, not the ones baked at compile (Makie `lines!`
    # and `scatter!` draw through here).
    rebakestages!(args, d.mesh, d.frag)

    lowered = loweredmesh(d, indices !== nothing)
    # The index buffer is the mesh stage's last argument (`GeometryAsMesh`),
    # converted like the others so the shader sees a device array.
    ix = indices === nothing ? nothing : bakearg(indices)
    mesh_bufs = ix === nothing ? buffer_types(d.mesh) :
        Tuple{map(argdevtype, d.mesh.device)..., argdevtype(ix)}
    c = compile_pipeline(d.device, lowered, d.color_formats, d.depth_format,
                         mesh_bufs, buffer_types(d.frag))

    enc = h.encoder
    setdrawstate!(enc, c)
    bind_stage!(enc, d.mesh, MTLm.set_mesh_bytes!, MTLm.MTLRenderStageMesh)
    if ix !== nothing
        useindices!(enc, indices, MTLm.MTLRenderStageMesh)
        # `scratchsize` covered only the stage's own arguments, so grow the
        # scratch to fit the index argument.
        length(d.mesh.scratch) < sizeof(typeof(ix)) &&
            resize!(d.mesh.scratch, sizeof(typeof(ix)))
        bind_arg!(MTLm.set_mesh_bytes!, enc, ix, length(d.mesh.device) + 1, d.mesh.scratch)
    end
    bind_stage!(enc, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)

    groups = Mantle.primitivecount(d.pipeline.topology, count)
    # No primitives (e.g. a one-point `lines`) draws nothing; Metal refuses a
    # zero grid.
    groups == 0 || drawmesh!(enc, groups, c.threads)
    return nothing
end

# Indirect lowered geometry draws are unsupported: `drawMeshThreadgroups` has no
# indirect form; that needs an `MTLIndirectCommandBuffer`.
Mantle.record_draw!(::MetalPassHandle, ::MetalCompiledGeometryDraw, _args,
                    n::Mantle.Commands; kw...) = error(
    "a geometry pipeline lowered onto a mesh stage cannot take a device-written " *
    "draw count yet: the count would be the number of mesh threadgroups, and " *
    "`drawMeshThreadgroups` reads its grid from the encoder, not from a buffer. " *
    "An `MTLIndirectCommandBuffer` is what does this on Metal. Draw with a host " *
    "count, or run the pipeline where the geometry stage is native.")
