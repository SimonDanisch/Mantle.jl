# Metal's answer to `src/vulkan/raytracing/hwtlas.jl`: the incremental
# acceleration structure a renderer builds a scene into, and the device-side
# `closest_hit` that traverses it.
#
# ── How a Julia kernel traverses on Metal ────────────────────────────────────
#
# On Vulkan `Raycore.closest_hit(::AdaptedAccel, ray)` lowers to an inline
# SPIR-V ray query, because Lava's emitter recognises the call. Metal has no
# such thing to emit: `intersector<>` is a C++ class template the Metal
# frontend instantiates, so no AIR symbol exists for it.
#
# What does exist is `MTLLinkedFunctions`. The traversal is written once in MSL
# (below), compiled at runtime with `MTLLibrary(dev, source)`, and LINKED into
# the pipeline of any Julia kernel that calls it. The kernel reaches it as an
# ordinary `ccall` to `__metal_linked_closest`, a name
# `GPUCompiler.isintrinsic` whitelists so IR validation accepts the otherwise
# undefined symbol.
#
# The acceleration structure travels through an **argument buffer**: a Julia
# kernel cannot declare an `instance_acceleration_structure` parameter, but it
# can pass a device pointer, and MSL can read the structure out of a struct in
# device memory. `AdaptedAccel.scene` carries that buffer.
#
# The net effect is that this file's `closest_hit` has the same signature,
# returns the same tuple, and is called from the same four sites in Hikari's
# `intersection.jl` as Vulkan's. Nothing in the integrator changes.

const HWTLAS_MSL = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct SceneAB { instance_acceleration_structure accel; };
// `inst` is the instance INDEX, which picks the instance's triangles; `user` is
// its CUSTOM index (the descriptor's `userID`), which Hikari reads as a
// per-instance material override. Two different numbers, and Vulkan has both.
struct HitOut  { float t; uint hit; uint prim; uint inst; float bu; float bv; uint user; };

// Returned BY VALUE. A struct of scalars matches what Julia's `ccall` expects on
// the same LLVM back-end, and it is the only shape that avoids giving every
// caller a scratch buffer to write hits into.
static inline HitOut pack(float tmax, thread const intersection_result<instancing, triangle_data> &res) {
    HitOut h;
    if (res.type == intersection_type::none) {
        // A miss reports tmax, matching the software traversal.
        h.t = tmax; h.hit = 0u; h.prim = 0u; h.inst = 0u; h.bu = 0.0f; h.bv = 0.0f;
        h.user = 0u;
    } else {
        h.t = res.distance; h.hit = 1u;
        h.prim = res.primitive_id; h.inst = res.instance_id;
        h.user = res.user_instance_id;
        float2 bc = res.triangle_barycentric_coord;
        h.bu = bc.x; h.bv = bc.y;
    }
    return h;
}

[[visible]] HitOut __metal_linked_closest(device const SceneAB *scene,
        float ox, float oy, float oz, float dx, float dy, float dz,
        float tmin, float tmax, uint cull)
{
    ray r; r.origin = float3(ox,oy,oz); r.direction = float3(dx,dy,dz);
    r.min_distance = tmin; r.max_distance = tmax;
    intersector<instancing, triangle_data> isect;
    return pack(tmax, isect.intersect(r, scene->accel, cull));
}

// `accept_any_intersection` is Metal's spelling of Vulkan's
// RayFlagsTerminateOnFirstHitKHR: stop at the first hit rather than the nearest.
[[visible]] HitOut __metal_linked_any(device const SceneAB *scene,
        float ox, float oy, float oz, float dx, float dy, float dz,
        float tmin, float tmax, uint cull)
{
    ray r; r.origin = float3(ox,oy,oz); r.direction = float3(dx,dy,dz);
    r.min_distance = tmin; r.max_distance = tmax;
    intersector<instancing, triangle_data> isect;
    isect.accept_any_intersection(true);
    return pack(tmax, isect.intersect(r, scene->accel, cull));
}
"""

"""The hit record the linked MSL functions return. Mirrors `HitOut` exactly."""
struct MetalHit
    t::Float32
    hit::UInt32
    prim::UInt32
    inst::UInt32
    bu::Float32
    bv::Float32
    user::UInt32
end
@assert sizeof(MetalHit) == 28 "MetalHit must match MSL's HitOut (28 bytes)"

# Compiled once per device. `MTLLibrary(dev, source)` runs the Metal frontend,
# which costs tens of milliseconds — fine once, not per scene.
const HWTLAS_FUNCS = Dict{Any,Nothing}()
const HWTLAS_FUNCS_LOCK = ReentrantLock()

function ensure_traversal_linked!(d::MetalDevice)
    key = pointer(d.dev)
    Base.@lock HWTLAS_FUNCS_LOCK begin
        haskey(HWTLAS_FUNCS, key) && return nothing
        lib = MTL.MTLLibrary(d.dev, HWTLAS_MSL)
        for name in ("__metal_linked_closest", "__metal_linked_any")
            Metal.register_linked_function!(d.dev, MTL.MTLFunction(lib, name))
        end
        HWTLAS_FUNCS[key] = nothing
    end
    return nothing
end

# Everything a traversal reads must be resident. Metal.jl already keeps a
# residency set per command queue; acceleration structures reached through an
# argument buffer are exactly the case where Metal cannot infer residency, and
# the failure mode without this is a silent miss rather than an error.
#
# Through Metal.jl's own bookkeeping rather than `add_allocation!` on the set: a
# set holds a strong reference to everything in it, and only an allocation
# Metal.jl knows about is taken back out when it is freed (`forget_resident!`).
function make_resident!(d::MetalDevice, resources...)
    Metal.install_queue_residency!(cmdqueue(d), d.dev)
    for r in resources
        r === nothing && continue
        Metal.make_persistently_resident!(r, d.dev)
    end
    return nothing
end

const DeviceU8Ptr = Core.LLVMPtr{UInt8, Metal.AS.Device}

# ── The device side ──────────────────────────────────────────────────────────
#
# Dispatch is on the ARRAY type, not on the accel: after `Adapt.adapt` the
# `hwtlas` field is `nothing` on every backend, so that parameter cannot tell
# them apart, but `MtlDeviceArray` vs `LavaDeviceArray` can.



@inline function _unpack(accel, h::MetalHit)
    if h.hit == UInt32(0)
        return (false, accel.empty, h.t, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    # Same indexing as Vulkan: per-instance offset into the flat triangle array,
    # plus the primitive index within that instance's BLAS.
    tri = hittriangle(accel, h.inst, h.prim)
    bary = SVector{3,Float32}(1f0 - h.bu - h.bv, h.bu, h.bv)
    # The fifth value is the instance CUSTOM index, not the instance index:
    # Vulkan's `gl_InstanceCustomIndexEXT`, Metal's `user_instance_id`, written
    # from the `userID` of the descriptor `build_accel!` builds. Hikari feeds it
    # to `resolve_mi_idx` as a per-instance medium-interface override, and a
    # `meshscatter!` bakes every face with interface 0 ("inherit") and puts each
    # sphere's material HERE. Returning 0 unconditionally, as this did while
    # Metal's descriptor carried no user ID, rendered every one of them black.
    # Returning `h.inst` instead would be the other mistake: it gives every
    # instance past the first the wrong interface.
    return (true, tri, h.t, bary, h.user)
end

# Two traversals, chosen by the accel's TYPE and not by a branch: `procedural`
# is `nothing` for a triangles-only scene, so the dispatch is static and a scene
# with no boxes compiles to exactly the call it always made.
#
# `mask` is the ray's cull mask, the intersector's mask argument: an instance is
# seen when it shares a bit with the instance's mask, which `writeinstances!`
# keeps to its low 8 bits, as Vulkan's is. Raycore's two-argument form passes
# `0xff`. `% UInt32` rather than a checked conversion, which would leave an error
# branch in every shading kernel.
#
# The procedural one costs more than a branch would: an `intersection_query`
# stops at every candidate and calls back into Julia, where `intersector<>`
# answers the whole ray in one go. A scene that has no use for it should not pay.
const MetalAdaptedTri  = AdaptedAccel{<:Any, <:Metal.MtlDeviceArray, <:Any, <:Any, <:Any, Nothing}
const MetalAdaptedProc = AdaptedAccel{<:Any, <:Metal.MtlDeviceArray, <:Any, <:Any, <:Any, <:Any}

@propagate_inbounds function Raycore.closest_hit(accel::MetalAdaptedTri, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    p = reinterpret(DeviceU8Ptr, pointer(accel.scene))
    h = ccall("extern __metal_linked_closest", llvmcall, MetalHit,
              (DeviceU8Ptr, Float32,Float32,Float32, Float32,Float32,Float32,
               Float32,Float32, UInt32),
              p, Float32(o[1]), Float32(o[2]), Float32(o[3]),
              Float32(d[1]), Float32(d[2]), Float32(d[3]),
              Float32(ray.t_min), Float32(ray.t_max), mask % UInt32)
    return _unpack(accel, h)
end

@propagate_inbounds function Raycore.closest_hit(accel::MetalAdaptedProc, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    p = reinterpret(DeviceU8Ptr, pointer(accel.scene))
    h = ccall("extern __metal_linked_closest_proc", llvmcall, MetalHit,
              (DeviceU8Ptr, Float32,Float32,Float32, Float32,Float32,Float32,
               Float32,Float32, UInt32),
              p, Float32(o[1]), Float32(o[2]), Float32(o[3]),
              Float32(d[1]), Float32(d[2]), Float32(d[3]),
              Float32(ray.t_min), Float32(ray.t_max), mask % UInt32)
    return _unpack_procedural(accel, h)
end

"""
A committed generated intersection, turned back into a primitive.

`procedural_commit` is what does it — the payload's own — and the best hit it is
given is rebuilt from the four floats the traversal carried back. Same tuple
shape and same primitive TYPE as a triangle hit, which is the whole point: a
caller of `closest_hit` never learns this ray met a box.
"""
@inline function _unpack_procedural(accel, h::MetalHit)
    if h.hit == UInt32(0)
        return (false, accel.empty, h.t, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    B = typeof(procedural_miss(accel.procedural))
    best = unpackbest(B, (h.t, h.bu, h.bv, _as_f32(h.inst)))
    prim = procedural_commit(accel.procedural, best, h.prim, h.t, accel.empty)
    return (true, prim, h.t, procedural_bary(accel.procedural, best), h.user)
end

@propagate_inbounds function Raycore.any_hit(accel::MetalAdaptedTri, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    p = reinterpret(DeviceU8Ptr, pointer(accel.scene))
    h = ccall("extern __metal_linked_any", llvmcall, MetalHit,
              (DeviceU8Ptr, Float32,Float32,Float32, Float32,Float32,Float32,
               Float32,Float32, UInt32),
              p, Float32(o[1]), Float32(o[2]), Float32(o[3]),
              Float32(d[1]), Float32(d[2]), Float32(d[3]),
              Float32(ray.t_min), Float32(ray.t_max), mask % UInt32)
    return _unpack(accel, h)
end

@propagate_inbounds function Raycore.any_hit(accel::MetalAdaptedProc, ray::Raycore.AbstractRay, mask::Integer)
    o = ray.o; d = ray.d
    p = reinterpret(DeviceU8Ptr, pointer(accel.scene))
    h = ccall("extern __metal_linked_any_proc", llvmcall, MetalHit,
              (DeviceU8Ptr, Float32,Float32,Float32, Float32,Float32,Float32,
               Float32,Float32, UInt32),
              p, Float32(o[1]), Float32(o[2]), Float32(o[3]),
              Float32(d[1]), Float32(d[2]), Float32(d[3]),
              Float32(ray.t_min), Float32(ray.t_max), mask % UInt32)
    return _unpack_procedural(accel, h)
end

# ── The host side ────────────────────────────────────────────────────────────

"""One batch of instances that all reference the same BLAS."""
const MetalInstanceBatch{Tri} =
    InstanceBatch{Tri,Int,MtlVector{Raycore.InstanceRecord}}

"""
    MetalHWTLAS{Tri} <: HWTLAS{Tri}

Metal's incremental acceleration structure. Same contract as `VulkanTLAS`:
`push!` geometry, `Raycore.sync!` to build, read the adapted form from
`static_tlas`.

Core owns geometry registration and batch mutations through `BatchedTLAS`.
This backend owns native builds, record packing, residency and traversal.
"""
mutable struct MetalHWTLAS{Tri} <: BatchedTLAS{Tri}
    backend::Any
    device::MetalDevice

    blas_list::Vector{MetalBLAS}
    # The order, the handles and the reindex on delete are core's
    # (`raytracing/batches.jl`); native types parameterize the shared batch.
    instances::Mantle.InstanceBatches{MetalInstanceBatch{Tri}}

    root_aabb::Raycore.Bounds3

    built::Any            # Union{Nothing, MetalTLAS{Tri}}
    tri_gpu::Any          # Union{Nothing, MtlVector{Tri}}
    off_gpu::Any          # Union{Nothing, MtlVector{UInt32}}
    # Every batch's records, in build order: what a hit reads its instance's
    # transform from (`AdaptedAccel.instances`). Written in place by a refit, so
    # the adapted form a plan holds sees it.
    records::Any          # Union{Nothing, MtlVector{Raycore.InstanceRecord}}
    scene_buf::Any        # Union{Nothing, MtlVector{UInt64}} — the argument buffer
    static_tlas::Any
    # Procedural traversal: the candidate's table, the pipeline that owns its
    # handle, and the adapted payload's bytes. All `nothing` for a triangles-only
    # scene, which is what makes `closest_hit` pick the cheaper traversal.
    proc_table::Any
    proc_pipeline::Any
    proc_payload::Any

    # The scene's procedural geometry, or `nothing` — beside `blas_list`
    # because that is what it is. Set by the caller that pushed an AABB BLAS
    # (`Hikari.push!(scene, ::FEMMaterial)`), read by `sync!` to build the
    # candidate's table, and carried to the kernel through `AdaptedAccel`.
    # Same field, same meaning, as the Vulkan side's.
    procedural::Any

    dirty::Bool
    transforms_dirty::Bool
end

HWTLAS{Tri}(backend::Metal.MetalBackend; kw...) where {Tri} = MetalHWTLAS{Tri}(backend; kw...)

function MetalHWTLAS{Tri}(backend::Metal.MetalBackend) where {Tri}
    d = Device(MetalAPI())
    ensure_traversal_linked!(d)
    MetalHWTLAS{Tri}(backend, d,
        MetalBLAS[],
        Mantle.InstanceBatches{MetalInstanceBatch{Tri}}(),
        Raycore.Bounds3(),
        nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
        true, false)
end

Raycore.wait_for_gpu!(t::MetalHWTLAS) = (Metal.synchronize(); t)

AdaptedAccel(t::MetalHWTLAS{Tri}) where {Tri} =
    AdaptedAccel(t, t.tri_gpu, t.off_gpu, Raycore.empty_triangle(Tri), t.scene_buf,
                 t.procedural, t.records)

function Adapt.adapt_structure(to, t::MetalHWTLAS)
    Raycore.sync!(t)
    return t.static_tlas::AdaptedAccel
end

# Passed as itself, a TLAS is held through the adapted form a kernel is handed: its
# own fields lead back to it through `static_tlas`, which the generic walker
# would follow forever.
Mantle.holdleaves!(holder, t::MetalHWTLAS) = Mantle.holdleaves!(holder, t.static_tlas)

"""Drop the host owner and adapt arrays plus the in-band native scene binding."""
Adapt.adapt_structure(to::Metal.Adaptor, accel::AdaptedAccel) =
    adaptaccel(to, accel, Base.Fix1(Adapt.adapt, to))

# ── Geometry ─────────────────────────────────────────────────────────────────

# Core supplies packed vertices; only native storage and AS creation live here.
# One native triangle per three consecutive vertices.
function buildgeometry!(t::MetalHWTLAS, vertices)
    vbuf = MTL.MTLBuffer(t.device.dev, sizeof(vertices); storage = Metal.SharedStorage)
    GC.@preserve vertices unsafe_copyto!(convert(Ptr{NTuple{3,Float32}}, MTL.contents(vbuf)),
                                        pointer(vertices), length(vertices))
    return build_blas(t.device, vbuf, length(vertices) ÷ 3)
end

geometryref(::MetalHWTLAS, i::Int) = i
instancearray(::MetalHWTLAS, records::AbstractVector) = MtlArray(records)
instancearray(::MetalHWTLAS, records::MtlArray) = records

# Preserve Metal's immediate record update and its return value. Submission and
# native build/refit policy remain backend-specific.
function updatebatch!(t::MetalHWTLAS, batch, transforms)
    batch.n == 0 || KI.Kernel(t.backend, update_instance_records_kernel!)(
        batch.records, transforms; ndrange = batch.n)
    return t
end

# ── The commit boundary ──────────────────────────────────────────────────────

"""
    mtlinstances_kernel!(descriptors, shading, records, offset, n, index, options, keep)

One batch's instances, from the records a caller writes into the two arrays a
build reads: `shading[offset + i]` is `records[i]` as it is (what a hit reads its
transform from), `descriptors[offset + i]` the descriptor Metal builds from,
naming the geometry at `index` in the structure's list.
"""
function mtlinstances_kernel!(descriptors, shading, records, offset::Int32, n::Int32,
                              index::UInt32, options::UInt32, keep::UInt32)
    i = KI.get_global_id().x
    i <= n || return nothing   # the launch is whole workgroups
    @inbounds r = records[i]
    j = offset + i
    @inbounds shading[j] = r
    @inbounds descriptors[j] = MetalInstanceDescriptor(r, index, options, keep)
    return nothing
end

"""
    writeinstances!(t, descriptors, shading)

Every batch's instances into `descriptors` and `shading`, by
[`mtlinstances_kernel!`](@ref), and wait for them: the acceleration-structure
build or refit after this goes out on a command buffer of its own, which nothing
orders behind Metal.jl's batch.
"""
function writeinstances!(t::MetalHWTLAS, descriptors::MtlArray, shading::MtlArray)
    options = UInt32(MTL.MTLAccelerationStructureInstanceOptionOpaque)
    offset = 0
    for b in t.instances
        b.n == 0 && continue
        KI.Kernel(t.backend, mtlinstances_kernel!)(descriptors, shading, b.records,
            Int32(offset), Int32(b.n), UInt32(b.blas - 1), options,
            b.hidden ? UInt32(0) : UInt32(0xff); ndrange = b.n)
        offset += b.n
    end
    KA.synchronize(t.backend)
    return nothing
end

function Raycore.sync!(t::MetalHWTLAS{Tri}) where {Tri}
    if !t.dirty && !t.transforms_dirty && t.static_tlas !== nothing
        return t
    end
    total = Mantle.ninstances(t.instances)

    # A TRANSFORM-only change is a REFIT, in place, keeping the structure and
    # therefore its `gpuResourceID` — which is the whole reason the structure is
    # built `refittable`. Rebuilding instead hands back a NEW
    # `MTLAccelerationStructure`, so `scene_buf` holds a new resource ID,
    # `static_tlas` is a new object, and anything holding the previous one keeps
    # tracing the previous geometry. Hikari's integrator does exactly that: it
    # caches the adapted scene and drops it only on `accel.dirty`, because a
    # refit writes the same buffers in place and a recorded plan survives it. So
    # a rebuild here makes `translate!` on a mesh produce a BYTE-IDENTICAL
    # image.
    if !t.dirty && t.built !== nothing && t.built.refittable &&
       t.built.count == total && t.static_tlas !== nothing
        # The structure and the shading records take the same transforms, both
        # written in place: recorded kernels retain the records' device view.
        writeinstances!(t, t.built.instances, t.records)
        refit_tlas!(t.device, t.built)
        t.transforms_dirty = false
        return t
    end
    # An EMPTY structure goes through the build like any other, and does not
    # return early: `adapt_structure` asserts `static_tlas::AdaptedAccel`, so a
    # sync that skips the build leaves a caller holding `nothing`. Empty means
    # every ray misses, which is a result and not a special case.
    #
    # One TLAS instance per record, each naming its batch's BLAS by index into
    # `blas_list`. A batch with no triangles gets `NOTRIANGLES` as its offset, so
    # its hits answer the empty triangle instead of the next batch's.
    metadata = Mantle.trianglemetadata(Tri, t.instances)

    # One element at least, as on Vulkan: a kernel binds these whatever the
    # scene holds.
    descriptors = MtlArray{MetalInstanceDescriptor}(undef, max(total, 1))
    t.records = MtlArray{Raycore.InstanceRecord}(undef, max(total, 1))
    writeinstances!(t, descriptors, t.records)
    t.built = build_accel!(t.device, t.blas_list, descriptors, total; refittable = true)
    t.tri_gpu = MtlArray(metadata.triangles)
    t.off_gpu = MtlArray(metadata.hits)

    # The argument buffer the kernel dereferences. One resource ID for a
    # triangles-only scene; three slots when there is procedural geometry, laid
    # out exactly as `SceneProcAB` in `procedural.jl` reads them —
    # `{accel, visible_function_table, device void *payload}`.
    if t.procedural === nothing
        t.scene_buf = MtlArray(reinterpret(UInt64, [t.built.handle.gpuResourceID]))
    else
        ensure_procedural_linked!(t.device)
        t.proc_payload = procedural_payload_bytes(t.procedural)
        t.proc_table, t.proc_pipeline =
            procedural_table(t.device, t.procedural, typeof(procedural_miss(t.procedural)))
        t.scene_buf = MtlArray(UInt64[
            reinterpret(UInt64, t.built.handle.gpuResourceID),
            reinterpret(UInt64, t.proc_table.gpuResourceID),
            # The GPU address, not `pointer`: a shader dereferences this and
            # the host pointer means nothing to it. Same verb `deviceaddress`
            # uses, and the same trap as any buffer reached by a stored address
            # -- it also has to be made resident, below.
            UInt64(t.proc_payload.data[].buffer.gpuAddress) +
                UInt64(t.proc_payload.offset * sizeof(UInt8)),
        ])
        # The table and the payload are reached through the argument buffer, so
        # Metal cannot infer their residency any more than it can the TLAS's.
        make_resident!(t.device, t.proc_table, t.proc_payload.data[].buffer)
        # …and everything the payload POINTS AT. Its bytes hold device addresses,
        # which Metal cannot follow for residency.
        procedural_resident!(t.device, t.procedural)
    end

    # Residency for everything traversal touches. Without this the structures
    # are not mapped for the dispatch and every ray misses, silently.
    make_resident!(t.device, t.built.handle, (b.handle for b in t.blas_list)...)

    t.dirty = false
    t.transforms_dirty = false
    t.static_tlas = AdaptedAccel(t)
    return t
end

# Metal traverses through a linked MSL function called from the shading kernel,
# so there is no shader binding table and no indirect trace-rays command. Still
# hardware traversal — see `supports_rt_pipeline` in `raytracing/api.jl`.
supports_rt_pipeline(::Metal.MetalBackend) = false
supports_rt_pipeline(::MetalHWTLAS) = false

# Metal traces AABB geometry: `closest_hit` above dispatches to the procedural
# traversal when the accel carries a payload. The loop is MSL -- an
# `intersection_query<>` is a C++ class template the frontend inlines, so it can
# be nothing else -- and the body is the caller's own `procedural_candidate`,
# compiled to an AIR visible function and reached through a table. See
# `src/metal/procedural.jl`.
supports_procedural_traversal(::Metal.MetalBackend) = true
# Every Apple-silicon GPU builds and traverses `MTLAccelerationStructure`s.
supports_hwtlas(::Metal.MetalBackend) = true
supports_hwtlas(::MetalDevice) = true
supports_procedural_traversal(::MetalHWTLAS) = true

# ── The three candidate verbs, inside a visible function ─────────────────────
#
# Overlays in Metal.jl's method table, as the Vulkan ones are in Lava's: a Mac
# compiles both trees, and plain methods would be one method defined twice.
#
# `procedural_candidate` is written against these as ZERO-argument calls, because
# on Vulkan they read an inline ray query that is already in scope. Here there is
# no query in scope: the loop is MSL (`intersection_query<>` is a C++ class
# template the frontend inlines, so it cannot be anything else) and the body is a
# Julia VISIBLE function it calls. The values arrive as parameters that
# `stage_builtins!` appends, which is the same mechanism `vertex_index()` uses.

@device_override @inline candidate_primitive_index() = Int(Metal.candidate_prim_raw()) + 1

@device_override @inline candidate_object_ray() =
    (Vec3f(Metal.candidate_ox(), Metal.candidate_oy(), Metal.candidate_oz()),
     Vec3f(Metal.candidate_dx(), Metal.candidate_dy(), Metal.candidate_dz()))

# A NO-OP here, and that is not a gap.
#
# The protocol says a payload calls this only for a `t` that IMPROVES on its
# running best, and then returns that improved best — so "the returned best
# improved" and "commit was called" are the same statement, and the MSL caller
# reads the first. It compares the `t` it gets back with the one it sent in and
# issues `commit_bounding_box_intersection` itself.
#
# Which is why there is no write-back channel: a visible function returns a value
# and has no other way out, and inventing one (a scratch slot per thread, say)
# would buy nothing the comparison does not already give.
@device_override @inline commit_intersection!(t) = nothing
