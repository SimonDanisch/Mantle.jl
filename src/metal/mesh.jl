# KernelInterface's mesh vocabulary, on Metal.
#
# `KernelInterface/src/mesh.jl` declares what a mesh stage may say and gives it
# a host implementation over `HostMeshOutput`; `Metal/src/compiler/mesh.jl`
# emits the AIR intrinsics. This file is the join: the portable verb on the one
# side, the `air.set_*_mesh` call on the other, so a mesh body compiles from one
# source on either backend.
#
# Two things the portable side fixes and this file has to undo, both because
# they are the right choices for a caller and the wrong ones for AIR:
#
#   * SLOTS COUNT FROM ONE, everywhere in KernelInterface — a vertex slot, a
#     primitive slot, an index into the vertices, the thread and group indices.
#     AIR counts every one of them from zero. The conversion lives here and
#     nowhere else; a shader body that had to remember which convention a
#     backend used would eventually get it wrong, and getting it wrong produces
#     a degenerate primitive rather than an error.
#   * A VERTEX IS A NAMEDTUPLE with `position` first, matching what a vertex
#     stage returns. AIR wants the position through its own intrinsic and each
#     remaining field through a numbered one, so the tuple is split by name at
#     compile time. The NUMBER is the declaration order of the mesh stage's
#     outputs, which is the same order `air_mesh_vertex_info` recorded in the
#     object's type — the two must agree or the fragment stage reads a
#     different field than the mesh stage wrote.
#
# `@device_override` and not a plain method: these are KernelInterface's
# functions, an extension may not redefine them, and a redefinition would also
# change the answer on the host where `HostMeshOutput` is the point.

# ── the writers ───────────────────────────────────────────────────────────────

"""
    mesh_field_index(V, name) -> Int32

Which numbered slot `name` occupies in AIR's per-vertex or per-primitive plane.

Zero-based and skipping `position`, because `air_mesh_vertex_info` numbers the
fields it emits `air.mesh_vertex_data` nodes for and `position` is not one of
them — it is `air.position`, which carries no index.
"""
@inline function mesh_field_index(@nospecialize(V::Type), name::Symbol)
    names = fieldnames(V)
    i = findfirst(==(name), names)
    i === nothing && error("`$name` is not a field of $V")
    # `position` is field 1 of a vertex tuple and absent from a primitive one,
    # so subtracting its presence gives the right base in both cases.
    return Int32(i - 1 - (first(names) === :position ? 1 : 0))
end

# The whole vertex, split by name. `@generated` because the split is a property
# of the TYPE and must cost nothing at run time: a mesh stage writes one of
# these per output vertex, and a per-field dictionary lookup in that loop would
# be the whole budget.
@generated function _write_mesh_vertex!(out::Metal.MeshPtr{V,P,NV,NP,T},
                                        slot::Int32, v::NamedTuple) where {V,P,NV,NP,T}
    calls = Expr[]
    hasfield(v, :position) ||
        error("a mesh vertex needs a `position`; got $(fieldnames(v))")
    # `flip_clip`, exactly as `MetalVertexStage` applies it to a vertex stage's
    # position. Mantle's clip space is Vulkan's (+y down) and Metal's is the
    # other one; a mesh stage that skipped the mirror would draw everything
    # upside down relative to the same geometry through a vertex stage, and
    # nothing in the pipeline would say so.
    push!(calls, :(Metal.set_position_mesh(out, slot,
                       flip_clip(_mesh_vec4(Base.getfield(v, :position))))))
    # A `Flat` output arrives HERE, in the vertex tuple, and is written to the
    # per-PRIMITIVE plane. Those are two different statements and both are
    # forced:
    #
    #   * The portable contract is Vulkan's, where flatness "says where a value
    #     is delivered, never what it is" (`vulkan/graphics/api.jl`) -- every
    #     declared output is written by `set_mesh_vertex!` and `Flat` only
    #     decorates the varying. A portable shader therefore writes its flat
    #     fields per vertex, and this backend does not get to ask for them
    #     somewhere else.
    #   * AIR has no flat qualifier on mesh vertex data; a mesh stage's flat
    #     varying IS per-primitive data (`air.mesh_primitive_data`), which is
    #     why `mesh_object_of` splits the declaration into two planes.
    #
    # Before this, the split was visible to the shader: `cell` was declared
    # `Flat`, so it was in `P` and not in `V`, and `mesh_field_index` raised
    # "`cell` is not a field of ..." from inside a `@generated` body. That does
    # not surface as the error it is -- the generator's throw becomes a runtime
    # call, so the stage failed to compile with `unsupported dynamic function
    # invocation` and a dozen `julia.gpu.state_getter` reports from the boxing
    # around it, naming neither `cell` nor `Flat`.
    flat = Symbol[]
    for n in fieldnames(v)
        n === :position && continue
        if hasfield(V, n)
            # FIELD before SLOT: that is AIR's order for the two data intrinsics
            # and not for `set_position_mesh` above. See
            # `Metal/src/compiler/mesh.jl` -- the two orders agree for the first
            # vertex and nowhere else, so getting it wrong leaves a shader whose
            # varyings are right at one corner.
            push!(calls, :(Metal.set_vertex_data_mesh(
                out, $(mesh_field_index(V, n)), slot,
                _mesh_component(Base.getfield(v, $(QuoteNode(n)))))))
        elseif hasfield(P, n)
            push!(flat, n)
        else
            error("`$n` is not an output of this mesh stage: it declares " *
                  "$(Base.tail(fieldnames(V))) smooth and $(fieldnames(P)) flat")
        end
    end
    if !isempty(flat)
        # The PROVOKING vertex, which is Vulkan's rule for which vertex a flat
        # varying takes its value from: the primitive's first. `slot` is 0-based
        # here, so primitive `slot ÷ vpp` is provoked by `slot % vpp == 0`.
        #
        # This reads the primitive off the vertex slot rather than off the index
        # list, so it is exact when a primitive owns its vertices -- which is
        # what a mesh stage that COMPUTES its geometry does, and what
        # `set_mesh_triangle!(out, t, 3t-2, 3t-1, 3t)` lays out. A stage that
        # shares one vertex between primitives has no single primitive to
        # attribute it to, and should write those outputs with the portable
        # `set_mesh_primitive_data!`, which names the primitive explicitly.
        vpp = mesh_verts_per_primitive(T)
        writes = [:(Metal.set_primitive_data_mesh(
                        out, $(mesh_field_index(P, n)), prim,
                        _mesh_component(Base.getfield(v, $(QuoteNode(n))))))
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

@generated function _write_mesh_primitive!(out::Metal.MeshPtr{V,P,NV,NP,T},
                                           slot::Int32, d::NamedTuple) where {V,P,NV,NP,T}
    calls = Expr[:(Metal.set_primitive_data_mesh(
                       out, $(mesh_field_index(P, n)), slot,
                       _mesh_component(Base.getfield(d, $(QuoteNode(n))))))
                 for n in fieldnames(d)]
    return Expr(:block, Expr(:meta, :inline), calls..., :(return nothing))
end

# What AIR takes: a plain tuple of floats, whatever the caller spelled it as.
# `Vec4f` and `NTuple{4,Float32}` are the same four floats and a shader may use
# either; the intrinsic declares one.
@inline _mesh_vec4(p::NTuple{4,Float32}) = p
@inline _mesh_vec4(p) = (Float32(p[1]), Float32(p[2]), Float32(p[3]), Float32(p[4]))
@inline _mesh_component(x::Union{Float32,NTuple{2,Float32},NTuple{3,Float32},
                                 NTuple{4,Float32}}) = x
@inline _mesh_component(x::Real) = Float32(x)
# `ntuple(..., Val(N))`, not `Tuple(f(c) for c in x)`. A generator is not statically
# unrolled, so the tuple it builds reaches the shader with an iteration protocol
# behind it — and the error path of that protocol asks for `kernel_state`, which a
# MESH stage does not have: `unsupported call to julia.gpu.state_getter`, at shader
# compile, naming none of this. `Val` makes every index a literal and the whole thing
# becomes N field reads. Same reason `Hikari.eval_poly` builds its monomials that way.
@inline _mesh_component(x) = ntuple(i -> Float32(@inbounds x[i]), Val(length(x)))

# No `@inline` in front of `@device_override`: the macro wraps a method
# DEFINITION, and a macrocall in that position registers an ordinary method
# instead of an overlay — which loads without complaint and then fails at shader
# compile with a `MethodError` on the portable verb. The bodies here are one
# call each and inline on their own.
Metal.@device_override KI.set_mesh_vertex!(out::Metal.MeshPtr, slot::Integer,
                                           v::NamedTuple) =
    _write_mesh_vertex!(out, Int32(slot) - Int32(1), v)

Metal.@device_override KI.set_mesh_primitive_data!(out::Metal.MeshPtr, slot::Integer,
                                                   d::NamedTuple) =
    _write_mesh_primitive!(out, Int32(slot) - Int32(1), d)

# ── the index writers ─────────────────────────────────────────────────────────
#
# AIR has ONE index intrinsic and addresses the index list flat: a triangle at
# primitive `p` occupies indices 3p, 3p+1, 3p+2. The arity a caller states is
# what turns a primitive slot into that offset, which is why KernelInterface has
# three verbs where AIR has one.
#
# The index itself is a `UInt8`, which is where `MAX_MESH_VERTICES = 256` comes
# from — a bound `MeshConfig` validates on the way in rather than truncating here.

@inline function _write_mesh_indices!(out::Metal.MeshPtr, base::Int32, idx::Vararg{Int32,N}) where {N}
    ntuple(Val(N)) do k
        Metal.set_index_mesh(out, base + Int32(k) - Int32(1),
                             unsafe_trunc(UInt8, idx[k] - Int32(1)))
    end
    return nothing
end

Metal.@device_override KI.set_mesh_triangle!(out::Metal.MeshPtr, slot::Integer,
                                             i0::Integer, i1::Integer, i2::Integer) =
    _write_mesh_indices!(out, (Int32(slot) - Int32(1)) * Int32(3),
                         Int32(i0), Int32(i1), Int32(i2))

Metal.@device_override KI.set_mesh_line!(out::Metal.MeshPtr, slot::Integer,
                                         i0::Integer, i1::Integer) =
    _write_mesh_indices!(out, (Int32(slot) - Int32(1)) * Int32(2), Int32(i0), Int32(i1))

Metal.@device_override KI.set_mesh_point!(out::Metal.MeshPtr, slot::Integer,
                                          i0::Integer) =
    _write_mesh_indices!(out, Int32(slot) - Int32(1), Int32(i0))

# ── the counts ────────────────────────────────────────────────────────────────

# Metal reads only the primitive count: how many VERTICES a threadgroup may
# write is bounded by the object's type, which is a compile-time constant. The
# portable verb takes both because SPIR-V's `OpSetMeshOutputsEXT` needs both, so
# the vertex count is deliberately dropped here rather than being absent from
# the interface.
#
# A threadgroup that declares ZERO primitives is eliminated whole by the driver,
# side effects included — which is the right answer for a geometry body that
# culled everything, and a trap for anything that hoped to write a buffer on the
# way past.
Metal.@device_override KI.set_mesh_outputs!(out::Metal.MeshPtr, ::Integer,
                                            nprimitives::Integer) =
    Metal.set_primitive_count_mesh(out, Int32(nprimitives))

# ── the builtins ──────────────────────────────────────────────────────────────
#
# Metal.jl's thread intrinsics already count from one, so these are a rename and
# a width, not a conversion. They are ordinary compute builtins in a mesh stage:
# `retag_stage!` threads them into the entry beside the object, and a mesh stage
# has threadgroup memory and `threadgroup_barrier` too.
Metal.@device_override KI.mesh_thread_index() =
    Int32(Metal.thread_index_in_threadgroup())
Metal.@device_override KI.mesh_group_index() =
    Int32(Metal.threadgroup_position_in_grid().x)

# `set_mesh_groups!` is deliberately absent: it belongs to an OBJECT stage, and
# Metal.jl emits no `air.object` program yet. A body that calls it gets a
# `MethodError` naming the verb, which is the accurate complaint — better than a
# method here that quietly did nothing.

# ── a portable MeshPipeline, compiled and drawn ──────────────────────────────
#
# Everything above is the DEVICE side: a body written in KernelInterface's verbs
# lowering to AIR. This is the host side — turning `Mantle.MeshPipeline`, which
# is a description and nothing else, into something Metal will draw.
#
# Two stages, no vertex stage and no input assembler. What a `GraphicsPipeline`
# gets from its `topology` a mesh pipeline gets from the mesh stage writing its
# own indices, which is why `MeshPipeline` has no `topology` field.

"""
    mesh_object_of(p::MeshPipeline) -> Type{<:MeshObject}

The output object type the mesh stage of `p` writes through.

Everything in it comes from the DECLARATION and nothing from the body: the two
planes are the `Flat` split of the stage's outputs, the bounds are its
`MeshConfig`'s, and the topology is what it says it emits. That is the whole
reason a mesh stage has to declare its outputs — the object's type is the
contract the fragment stage is linked against, and it has to exist before either
stage is compiled.
"""
function mesh_object_of(p::Mantle.MeshPipeline)
    cfg = Mantle.meshconfig(p)
    outs = Mantle.stageoutputs(p.mesh)
    vals = Mantle.valuetypes(outs)
    smooth = Mantle.smoothoutputs(p.mesh)
    flat   = Mantle.flatoutputs(p.mesh)
    # `position` first and always, exactly as `outputtype` prepends it for a
    # vertex stage: every mesh vertex has one and no declaration mentions it.
    V = NamedTuple{(:position, smooth...),
                   Tuple{NTuple{4,Float32}, (vals[n] for n in smooth)...}}
    P = NamedTuple{flat, Tuple{(vals[n] for n in flat)...}}
    return Metal.MeshObject{V, P, cfg.max_vertices, cfg.max_primitives,
                            mesh_topology_symbol(cfg.topology)}
end

"""
    mesh_topology_symbol(topology) -> Symbol

What AIR calls the KIND of primitive a mesh stage emits.

Metal has three — triangle, line, point — where Mantle has five, because a
STRIP is a property of how an assembler groups an input stream and a mesh stage
has no input stream. A strip topology reaches the emitter, which turns it into
explicitly indexed primitives of the same kind; by the time the object's type is
built, only the kind is left.
"""
mesh_topology_symbol(::Union{Mantle.TriangleList,Mantle.TriangleStrip}) = :triangle
mesh_topology_symbol(::Union{Mantle.LineList,Mantle.LineStrip})         = :line
mesh_topology_symbol(::Mantle.PointList)                                = :point

"""
    mesh_verts_per_primitive(kind::Symbol) -> Int32

How many vertices one primitive of AIR's `kind` has.

On the SYMBOL and not on a `Mantle` topology, because its caller is
`_write_mesh_vertex!`, which has only the `MeshObject`'s last type parameter to
go on -- by then the strip/list distinction is gone and only the kind is left.
"""
mesh_verts_per_primitive(kind::Symbol) =
    kind === :triangle ? Int32(3) :
    kind === :line     ? Int32(2) :
    kind === :point    ? Int32(1) :
    error("unknown mesh primitive kind `$kind`")

"""
    mesh_stage_lead(p::MeshPipeline) -> Tuple

What the mesh body is handed before any caller argument: the object it writes
its vertices and primitives through.

One function because two places need it and must agree: `compile_pipeline`
compiles the body at `(lead..., buffers...)`, and `vertextouches` walks it at the
same signature to learn what the draw's arguments are used for. A walk at any
other signature describes a function nobody compiles.
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

const MESH_CACHE = Dict{Any,Any}()
const MESH_CACHE_LOCK = ReentrantLock()

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
    compile_pipeline(p::MeshPipeline, color_formats, depth_format, mesh_bufs, frag_bufs)

Compile a portable [`MeshPipeline`](@ref) into a Metal render pipeline state.

`mesh_bufs`/`frag_bufs` are the Tuple types of each stage's BUFFER arguments, the
same contract `compile_pipeline` for a `GraphicsPipeline` has. The mesh stage's
object is prepended here rather than named by the caller: it is not a buffer, the
caller does not bind it, and its type is the pipeline's to derive.

Cached on everything baked in, as the graphics one is.
"""
function compile_pipeline(p::Mantle.MeshPipeline,
                          color_formats::Vector{MTLm.MTLPixelFormat},
                          depth_format::Union{Nothing,MTLm.MTLPixelFormat},
                          mesh_bufs::Type, frag_bufs::Type)
    p.object === nothing ||
        error("this backend has no object stage yet: Metal.jl emits no `air.object` " *
              "program, so a MeshPipeline's mesh threadgroups are dispatched by the " *
              "host. Build the pipeline without one and pass the group count to `draw!`.")

    key = (p.mesh, p.fragment, mesh_bufs, frag_bufs, color_formats, depth_format,
           typeof(p.blend), typeof(p.cull), typeof(p.depth))
    Base.@lock MESH_CACHE_LOCK begin
        cached = get(MESH_CACHE, key, nothing)
        cached === nothing || return cached::MetalCompiledMeshPipeline
    end

    dev = Metal.device()
    cfg = Mantle.meshconfig(p)
    Obj = mesh_object_of(p)

    # A wrapper only to splat the packed arguments (see `PackedArgs`): a mesh
    # stage already writes through the object it is handed, which is the AIR
    # shape, so there is nothing else to translate.
    mesh_tt = Tuple{mesh_stage_lead(p)..., mesh_bufs.parameters...}
    mfn = MetalMeshStage{typeof(Mantle.stagefunction(p.mesh))}()
    mfun, mlib = compile_stage_function(mfn, mesh_tt, :mesh,
                                        string(nameof(Mantle.stagefunction(p.mesh))) * "_ms")

    isempty(color_formats) &&
        error("a mesh pipeline needs a colour attachment: Metal has no " *
              "depth-only mesh pipeline, and a nil fragment function is refused " *
              "for one.")
    VIn  = varying_type(p)
    FOut = NamedTuple{ntuple(i -> Symbol(:color, i), length(color_formats)),
                      NTuple{length(color_formats), NTuple{4,Float32}}}
    ntex = Mantle.ntextures(p.fragment)
    ffn = MetalFragmentStage{typeof(Mantle.stagefunction(p.fragment)), VIn, FOut, ntex}()
    frag_tt = Tuple{frag_bufs.parameters..., varying_markers(VIn)...,
                    texture_markers(ntex)..., Core.LLVMPtr{FOut,1}}
    ffun, flib = compile_stage_function(ffn, frag_tt, :fragment,
                                        string(nameof(Mantle.stagefunction(p.fragment))) * "_fs")

    desc = MTLm.MTLMeshRenderPipelineDescriptor()
    desc.meshFunction = mfun
    desc.fragmentFunction = ffun
    for (i, fmt) in enumerate(color_formats)
        att = desc.colorAttachments[i]
        att.pixelFormat = fmt
        apply_blend!(att, p.blend)
    end
    depth_format === nothing || (desc.depthAttachmentPixelFormat = depth_format)
    # The declared threadgroup width. Metal compares this against what the shader
    # says and refuses a mismatch, which is why it comes from the `MeshConfig`
    # rather than from the draw.
    desc.maxTotalThreadsPerMeshThreadgroup = cfg.threads
    state = MTLm.MTLRenderPipelineState(dev, desc)

    dstate = nothing
    if depth_format !== nothing && !(p.depth isa Mantle.DepthOff)
        dd = MTLm.MTLDepthStencilDescriptor()
        dd.depthCompareFunction = mtl_depth_compare(p.depth)
        dd.depthWriteEnabled    = depth_writes(p.depth)
        dstate = MTLm.MTLDepthStencilState(dev, dd)
    end

    compiled = MetalCompiledMeshPipeline(state, dstate, mtl_cull(p.cull), cfg.threads,
                                         Any[mlib, flib])
    Base.@lock MESH_CACHE_LOCK begin
        MESH_CACHE[key] = compiled
    end
    return compiled
end

# `draw!` is CORE's, in `src/graphics/record.jl`. This backend used to shadow it
# with a method of its own taking `buffers=`/`frag_buffers=` and explicit
# `vert_tt`/`frag_tt`, which meant the "portable entry point" its docstring
# claimed to be was portable in name only -- a caller spelled the same draw two
# ways depending on the backend. Core builds on `compile_draw`, `pass!` and
# `record_draw!`, all of which this backend already answers, and infers the
# shader argument types from the arrays it is handed rather than being told.


# Metal.jl emits `air.mesh` and the vocabulary above lowers onto it, so the
# capability answers `true` — and it answers for the BACKEND and the device
# alike, because a caller may hold either.
Mantle.supports_mesh_pipeline(::Metal.MetalBackend) = true
Mantle.supports_mesh_pipeline(::MetalDevice) = true

# ── a geometry pipeline, run as a mesh pipeline ──────────────────────────────
#
# `Mantle.lower_geometry_to_mesh` rewrites the vertex + geometry pair as one mesh
# stage; these two hooks are what make that automatic, so a renderer hands this
# backend the same `GraphicsPipeline` it hands Vulkan and does not ask which
# stages exist here. That is the whole point of the lowering living in core:
# `supports_geometry_stage` answers `false` and a geometry pipeline still draws.
#
# Dispatch on the pipeline's geometry parameter rather than a branch, because
# `GraphicsPipeline` carries it in its type: `Nothing` or a `GeometryShader`.

"""One draw of a lowered geometry pipeline: the description, and the baked arguments."""
mutable struct MetalCompiledGeometryDraw
    const pipeline::Mantle.GraphicsPipeline
    const color_formats::Vector{MTLm.MTLPixelFormat}
    const depth_format::Union{Nothing,MTLm.MTLPixelFormat}
    const mesh::StageArgs
    const frag::StageArgs
    # The lowered mesh pipeline, remembered per INDEXEDNESS — the one thing about
    # it that is the draw's rather than the pipeline's. `lower_geometry_to_mesh`
    # is a pure function of the two and was being called again for every draw of
    # every frame: 47 KB and 836 allocations a frame on a figure with a `lines!`
    # and a `scatter!`, for a `MeshPipeline` identical to the last one.
    #
    # Here rather than in a global table beside the function, because this is
    # where the answer's inputs already live and a draw is exactly its scope.
    lowered_plain::Any
    lowered_indexed::Any
end

MetalCompiledGeometryDraw(p, cf, df, mesh, frag) =
    MetalCompiledGeometryDraw(p, cf, df, mesh, frag, nothing, nothing)

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

"""What `compile_draw` hands back for a `MeshPipeline`, and `record_draw!` binds.

The mesh counterpart to [`MetalCompiledDraw`](@ref). It exists for the same reason
that one does: core's `draw!` compiles once and records per frame, so the pipeline
and the two stages' baked arguments have to outlive the pass.

A `MeshPipeline` has no vertex stage and no index buffer -- the mesh stage writes
its own primitive indices -- so `count` is a THREADGROUP count, not a vertex count,
and there is nothing here for `indices` to mean.
"""
struct MetalCompiledMeshDraw
    pipeline::MetalCompiledMeshPipeline
    mesh::StageArgs
    frag::StageArgs
end

function Mantle.compile_draw(d::MetalDevice, p::Mantle.MeshPipeline,
                             color_formats, depth_format, mesh_args, frag_args;
                             bindings = nothing)
    mesh = StageArgs(mesh_args)
    frag = StageArgs(frag_args)
    cfmts = MTLm.MTLPixelFormat[mtlformat(T) for T in color_formats]
    dfmt = depth_format === nothing ? nothing : mtlformat(depth_format)
    pipeline = compile_pipeline(p, cfmts, dfmt, buffer_types(mesh), buffer_types(frag))
    return MetalCompiledMeshDraw(pipeline, mesh, frag)
end

"""
    record_draw!(h, d::MetalCompiledMeshDraw, args, groups; instances, indices)

Record one mesh draw: `groups` threadgroups of the width the pipeline was compiled
at.

`indices` is refused rather than ignored. A mesh stage emits its own primitive
indices through `set_mesh_outputs!`, so an index buffer here is a caller who
believes something about this draw that is not true, and silently dropping it
would draw the right picture for the wrong reason until the day it did not.
"""
function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledMeshDraw, args, groups;
                             instances::Integer = 1, indices = nothing)
    indices === nothing || throw(ArgumentError(
        "record_draw!: a mesh pipeline takes no index buffer — the mesh stage writes " *
        "its own primitive indices. `count` is its threadgroup count."))
    instances == 1 || throw(ArgumentError(
        "record_draw!: this backend dispatches mesh threadgroups directly, so there " *
        "is no instance count to multiply them by; got instances = $(instances)."))
    isempty(d.mesh.device) || rebake!(d.mesh, args)
    isempty(d.frag.device) || rebake!(d.frag, args)
    c = d.pipeline
    MTLm.set_pipeline!(h.encoder, c.state)
    c.depth_state === nothing ||
        MTLm.set_depth_stencil_state!(h.encoder, c.depth_state)
    MTLm.set_cull_mode!(h.encoder, c.cull)
    # Counter-clockwise, for the reason the vertex path sets it: `flip_clip` mirrors
    # y and a mirror reverses the handedness of every primitive.
    MTLm.set_front_facing_winding!(h.encoder, MTLm.MTLWindingCounterClockwise)
    bind_stage!(h.encoder, d.mesh, MTLm.set_mesh_bytes!, MTLm.MTLRenderStageMesh)
    bind_stage!(h.encoder, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)
    MTLm.draw_mesh_threadgroups!(h.encoder, MTLm.MTLSize(groups, 1, 1),
                                            MTLm.MTLSize(1, 1, 1),
                                            MTLm.MTLSize(c.threads, 1, 1))
    return nothing
end

"""
    compile_draw(d::MetalDevice, p::GeometryPipeline, …)

Bake a geometry pipeline's arguments and keep what its mesh pipeline needs.

The PIPELINE is not compiled here, and the reason is the index buffer. A lowered
mesh stage reads it as one of its own arguments — a mesh pipeline has no index
buffer, so the fetch a vertex stage got for free has to happen in the shader —
and whether a draw is indexed is the DRAW's business, arriving at
[`record_draw!`](@ref). So the stage's signature is only complete there, and this
keeps the two `StageArgs` and the formats until it is.

The vertex body is checked for its `VertexIndex` parameter here rather than there,
because a missing one is a mistake in the SHADER and should be reported the first
time the pipeline is used, not from inside a shader compilation.
"""
function Mantle.compile_draw(d::MetalDevice,
                             p::Mantle.GeometryPipeline,
                             color_formats, depth_format, vert_args, frag_args;
                             bindings = nothing)
    mesh = StageArgs(vert_args)
    frag = StageArgs(frag_args)
    Mantle.requirevertexindex(Mantle.stagefunction(p.vertex), bodytypes(buffer_types(mesh)))
    cfmts = MTLm.MTLPixelFormat[mtlformat(T) for T in color_formats]
    dfmt = depth_format === nothing ? nothing : mtlformat(depth_format)
    return MetalCompiledGeometryDraw(p, cfmts, dfmt, mesh, frag)
end

"""
    record_draw!(h, d::MetalCompiledGeometryDraw, args, count; instances, indices)

Record a lowered geometry draw: one mesh threadgroup per input primitive.

`count` is the draw's vertex or index count, exactly as the geometry pipeline
would take it, and how many primitives an assembler would have made of it is what
`Mantle.primitivecount` answers — so the caller counts vertices whichever backend
it is on.
"""
function Mantle.record_draw!(h::MetalPassHandle, d::MetalCompiledGeometryDraw, args,
                             count::Integer; instances::Integer = 1, indices = nothing)
    instances == 1 || error(
        "a lowered geometry pipeline draws one instance, and this draw asks for " *
        "$instances. A mesh dispatch has a three-dimensional threadgroup grid and " *
        "the instance would go on its second axis, which needs a portable name for " *
        "`instance_index()` inside a mesh stage; there is none yet and nothing in " *
        "tree instances a geometry pipeline.")

    # Same as the vertex path: bind what THIS frame handed over, not what was
    # baked at compile. A Makie `lines!` or `scatter!` is a geometry pipeline and
    # so comes through here rather than through `MetalCompiledDraw`, which is why
    # rebinding a camera moved everything on screen except the plots.
    isempty(d.mesh.device) || rebake!(d.mesh, args)
    isempty(d.frag.device) || rebake!(d.frag, args)

    lowered = loweredmesh(d, indices !== nothing)
    # The index buffer as the mesh stage's LAST argument, which is where
    # `GeometryAsMesh` reads it. Converted the same way every other argument is,
    # so what the shader sees is the device array and not a raw address.
    ix = indices === nothing ? nothing : bakearg(indices)
    mesh_bufs = ix === nothing ? buffer_types(d.mesh) :
        Tuple{map(argdevtype, d.mesh.device)..., argdevtype(ix)}
    c = compile_pipeline(lowered, d.color_formats, d.depth_format,
                         mesh_bufs, buffer_types(d.frag))

    enc = h.encoder
    MTLm.set_pipeline!(enc, c.state)
    c.depth_state === nothing || MTLm.set_depth_stencil_state!(enc, c.depth_state)
    MTLm.set_cull_mode!(enc, c.cull)
    # As the vertex path does, and for the same reason: `flip_clip` mirrors y,
    # which reverses the handedness of every primitive the emitter wound.
    MTLm.set_front_facing_winding!(enc, MTLm.MTLWindingCounterClockwise)
    bind_stage!(enc, d.mesh, MTLm.set_mesh_bytes!, MTLm.MTLRenderStageMesh)
    if ix !== nothing
        ibuf = metal_buffer(indices)
        ibuf === nothing && error(
            "an indexed draw needs a device buffer of indices, got $(typeof(indices))")
        # Resident for the MESH stage, not the vertex one: on this path the
        # indices are read by a shader, and an address the encoder was never told
        # about reads as zeros rather than faulting.
        MTLm.use!(enc, ibuf, MTLm.ReadUsage, MTLm.MTLRenderStageMesh)
        # The mesh stage's scratch: the index buffer is one more of its
        # arguments, so it stages the same way the rest do. Sized explicitly
        # because `scratchsize` only ever saw the stage's OWN arguments, and an
        # index buffer being narrower than a 4x4 matrix is luck, not a contract.
        length(d.mesh.scratch) < sizeof(typeof(ix)) &&
            resize!(d.mesh.scratch, sizeof(typeof(ix)))
        bind_arg!(MTLm.set_mesh_bytes!, enc, ix, length(d.mesh.device) + 1, d.mesh.scratch)
    end
    bind_stage!(enc, d.frag, MTLm.set_fragment_bytes!, MTLm.MTLRenderStageFragment)

    groups = Mantle.primitivecount(d.pipeline.topology, count)
    # Not an error: a `lines` plot with one point assembles no primitive, and a
    # draw of nothing is a draw of nothing. `drawMeshThreadgroups` with a zero
    # grid is refused by Metal.
    groups == 0 && return nothing
    MTLm.draw_mesh_threadgroups!(enc, MTLm.MTLSize(groups, 1, 1),
                                      MTLm.MTLSize(1, 1, 1),
                                      MTLm.MTLSize(c.threads, 1, 1))
    return nothing
end

# An indirect draw of a lowered geometry pipeline. The threadgroup count would
# have to be read out of the same buffer, and `drawMeshThreadgroups` has no
# indirect form that takes one — `MTLIndirectCommandBuffer` is the mechanism, and
# it is a different object with its own encoding path.
Mantle.record_draw!(::MetalPassHandle, ::MetalCompiledGeometryDraw, _args,
                    n::Mantle.Commands; kw...) = error(
    "a geometry pipeline lowered onto a mesh stage cannot take a device-written " *
    "draw count yet: the count would be the number of mesh threadgroups, and " *
    "`drawMeshThreadgroups` reads its grid from the encoder, not from a buffer. " *
    "An `MTLIndirectCommandBuffer` is what does this on Metal. Draw with a host " *
    "count, or run the pipeline where the geometry stage is native.")
