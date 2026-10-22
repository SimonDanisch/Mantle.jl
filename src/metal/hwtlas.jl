# Metal's incremental hardware TLAS and its device-side traversal. `intersector<>`
# is a C++ template with no AIR symbol, so traversal is MSL (`HWTLAS_MSL`) linked
# into each calling kernel through `MTLLinkedFunctions` and reached by an extern
# `ccall` (Metal.jl's `isintrinsic` accepts the `__metal_linked_` prefix). A Julia
# kernel cannot take an `instance_acceleration_structure`, so the TLAS travels in
# an argument buffer, `AdaptedAccel.scene`.

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

"""The hit record the linked MSL functions return; mirrors `HitOut` field for field."""
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

# Compiled once per device: `MTLLibrary(dev, source)` runs the Metal frontend,
# tens of milliseconds.
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

# Metal cannot infer residency through an argument buffer, and a non-resident
# structure makes rays miss silently. Through Metal.jl's per-queue residency set
# rather than `add_allocation!`: the set holds strong references, and only
# allocations Metal.jl tracks leave it when freed (`forget_resident!`).
function make_resident!(d::MetalDevice, resources...)
    Metal.install_queue_residency!(cmdqueue(d), d.dev)
    for r in resources
        r === nothing && continue
        Metal.make_persistently_resident!(r, d.dev)
    end
    return nothing
end

const DeviceU8Ptr = Core.LLVMPtr{UInt8, Metal.AS.Device}

@inline function unpack(accel, h::MetalHit)
    if h.hit == UInt32(0)
        return (false, accel.empty, h.t, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    # As on Vulkan: per-instance offset into the flat triangle array plus the
    # primitive index within the instance's BLAS.
    tri = hittriangle(accel, h.inst, h.prim)
    bary = SVector{3,Float32}(1f0 - h.bu - h.bv, h.bu, h.bv)
    # Fifth value: the instance's custom index (descriptor `userID`, Vulkan's
    # `gl_InstanceCustomIndexEXT`), not its index. Hikari's `resolve_mi_idx` reads
    # it as a per-instance medium-interface override, where `meshscatter!` puts
    # each instance's material; 0 or `h.inst` here renders the wrong material.
    return (true, tri, h.t, bary, h.user)
end

# Dispatch on the triangle array type (`hwtlas` is `nothing` on every backend
# after `Adapt.adapt`) and statically on `procedural`: `nothing` takes the cheaper
# `intersector<>` call instead of a query that calls back into Julia per candidate.
# `mask % UInt32` avoids an error branch in every kernel.
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
    return unpack(accel, h)
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
    return unpack_procedural(accel, h)
end

"""
    unpack_procedural(accel, h::MetalHit)

A committed procedural hit as a `closest_hit` tuple. The payload's
`procedural_commit` rebuilds the primitive from the best hit's four floats, so
the tuple shape and primitive type match a triangle hit.
"""
@inline function unpack_procedural(accel, h::MetalHit)
    if h.hit == UInt32(0)
        return (false, accel.empty, h.t, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    B = typeof(procedural_miss(accel.procedural))
    best = unpackbest(B, (h.t, h.bu, h.bv, as_f32(h.inst)))
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
    return unpack(accel, h)
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
    return unpack_procedural(accel, h)
end

"""One batch of instances that all reference the same BLAS."""
const MetalInstanceBatch{Tri} =
    InstanceBatch{Tri,Int,MtlVector{Raycore.InstanceRecord}}

"""
    MetalHWTLAS{Tri} <: BatchedTLAS{Tri}

Metal's incremental acceleration structure, with `VulkanTLAS`'s contract: `push!`
geometry, `Raycore.sync!` to build, read the adapted form from `static_tlas`.
Core's `BatchedTLAS` owns registration and batch mutation; this type owns native
builds, record packing, residency and traversal.
"""
mutable struct MetalHWTLAS{Tri} <: BatchedTLAS{Tri}
    backend::Any
    device::MetalDevice

    blas_list::Vector{MetalBLAS}
    # Order, handles and reindex on delete are core's (`raytracing/batches.jl`).
    instances::Mantle.InstanceBatches{MetalInstanceBatch{Tri}}

    root_aabb::Raycore.Bounds3

    built::Any            # Union{Nothing, MetalTLAS{Tri}}
    tri_gpu::Any          # Union{Nothing, MtlVector{Tri}}
    off_gpu::Any          # Union{Nothing, MtlVector{UInt32}}
    # Every batch's records in build order, read by hits for the instance
    # transform (`AdaptedAccel.instances`). A refit writes them in place, so an
    # adapted form a plan holds sees the change.
    records::Any          # Union{Nothing, MtlVector{Raycore.InstanceRecord}}
    scene_buf::Any        # Union{Nothing, MtlVector{UInt64}}: the argument buffer
    static_tlas::Any
    # Procedural traversal: the candidate's function table, the pipeline that
    # owns its handle, and the payload bytes. `nothing` for a triangles-only scene.
    proc_table::Any
    proc_pipeline::Any
    proc_payload::Any

    # The scene's procedural geometry or `nothing`, which selects the
    # triangles-only traversal. Set by the caller that pushes an AABB BLAS
    # (Hikari's `push!(scene, mesh, ::FEMMaterial)`), read by `sync!` to build
    # the candidate's table, passed to kernels through `AdaptedAccel`.
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

# Held through its adapted form: `static_tlas` leads back to `t`, so the generic
# walker would recurse forever.
Mantle.holdleaves!(holder, t::MetalHWTLAS) = Mantle.holdleaves!(holder, t.static_tlas)

"""Adapt the arrays and the in-band scene buffer, dropping the host `hwtlas` owner."""
Adapt.adapt_structure(to::Metal.Adaptor, accel::AdaptedAccel) =
    adaptaccel(to, accel, Base.Fix1(Adapt.adapt, to))

# `vertices` comes packed from core, three consecutive vertices per triangle.
function buildgeometry!(t::MetalHWTLAS, vertices)
    vbuf = MTL.MTLBuffer(t.device.dev, sizeof(vertices); storage = Metal.SharedStorage)
    GC.@preserve vertices unsafe_copyto!(convert(Ptr{NTuple{3,Float32}}, MTL.contents(vbuf)),
                                        pointer(vertices), length(vertices))
    return build_blas(t.device, vbuf, length(vertices) ÷ 3)
end

geometryref(::MetalHWTLAS, i::Int) = i
instancearray(::MetalHWTLAS, records::AbstractVector) = MtlArray(records)
instancearray(::MetalHWTLAS, records::MtlArray) = records

# Writes the transforms into the batch's records on the device at once; the
# refit happens at `sync!`.
function updatebatch!(t::MetalHWTLAS, batch, transforms)
    batch.n == 0 || KI.Kernel(t.backend, update_instance_records_kernel!)(
        batch.records, transforms; ndrange = batch.n)
    return t
end

"""
    mtlinstances_kernel!(descriptors, shading, records, offset, n, index, options, keep)

One batch: `shading[offset + i] = records[i]` (what a hit reads its transform
from) and `descriptors[offset + i]`, the descriptor Metal builds from, naming
geometry `index` (zero-based) in the structure's list.
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

Write every batch's instances with [`mtlinstances_kernel!`](@ref) and wait: the
build or refit that follows runs on a command buffer of its own, which nothing
orders after Metal.jl's batch.
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

    # A transform-only change refits in place, keeping the structure and its
    # `gpuResourceID` (why it is built `refittable`). A rebuild makes a new
    # structure and `static_tlas`, while Hikari keeps tracing its cached adapted
    # scene until `accel.dirty`, so a moved mesh would render unchanged.
    if !t.dirty && t.built !== nothing && t.built.refittable &&
       t.built.count == total && t.static_tlas !== nothing
        # Structure and shading records take the same transforms in place:
        # recorded kernels retain the records' device view.
        writeinstances!(t, t.built.instances, t.records)
        refit_tlas!(t.device, t.built)
        t.transforms_dirty = false
        return t
    end
    # An empty scene is built too (every ray misses): `adapt_structure` asserts
    # `static_tlas::AdaptedAccel`. One TLAS instance per record, naming its
    # batch's BLAS by index into `blas_list`; a batch without triangles gets offset
    # `NOTRIANGLES`, so its hits answer the empty triangle, not the next batch's.
    metadata = Mantle.trianglemetadata(Tri, t.instances)

    # At least one element, as on Vulkan: kernels bind these whatever the scene holds.
    descriptors = MtlArray{MetalInstanceDescriptor}(undef, max(total, 1))
    t.records = MtlArray{Raycore.InstanceRecord}(undef, max(total, 1))
    writeinstances!(t, descriptors, t.records)
    t.built = build_accel!(t.device, t.blas_list, descriptors, total; refittable = true)
    t.tri_gpu = MtlArray(metadata.triangles)
    t.off_gpu = MtlArray(metadata.hits)

    # The argument buffer the kernel dereferences: the TLAS resource ID, or with
    # procedural geometry `SceneProcAB`'s layout in `procedural.jl`,
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
            # The GPU address, not `pointer`: the shader dereferences it. Like
            # any buffer reached by a stored address it must be made resident.
            UInt64(t.proc_payload.data[].buffer.gpuAddress) +
                UInt64(t.proc_payload.offset * sizeof(UInt8)),
        ])
        # Reached through the argument buffer, so Metal cannot infer residency.
        make_resident!(t.device, t.proc_table, t.proc_payload.data[].buffer)
        # Also whatever the device addresses in the payload point at.
        procedural_resident!(t.device, t.procedural)
    end

    # Non-resident structures are not mapped for the dispatch: every ray misses silently.
    make_resident!(t.device, t.built.handle, (b.handle for b in t.blas_list)...)

    t.dirty = false
    t.transforms_dirty = false
    t.static_tlas = AdaptedAccel(t)
    return t
end

# Traversal is a linked MSL function called from the shading kernel: hardware
# traversal, but no shader binding table or indirect trace-rays command.
supports_rt_pipeline(::Metal.MetalBackend) = false
supports_rt_pipeline(::MetalHWTLAS) = false

# AABB geometry: an MSL `intersection_query<>` loop whose body is the caller's
# `procedural_candidate`, compiled to an AIR visible function and reached through
# a table (`procedural.jl`).
supports_procedural_traversal(::Metal.MetalBackend) = true
# Every Apple-silicon GPU builds and traverses `MTLAccelerationStructure`s.
supports_hwtlas(::Metal.MetalBackend) = true
supports_hwtlas(::MetalDevice) = true
supports_procedural_traversal(::MetalHWTLAS) = true

# The candidate verbs inside a procedural visible function. Overlays in Metal.jl's
# method table, as Vulkan's are in Lava's: a Mac compiles both, so plain methods
# would be defined twice. The values arrive as parameters `stage_builtins!`
# appends, as for `vertex_index()`.

@device_override @inline candidate_primitive_index() = Int(Metal.candidate_prim_raw()) + 1

@device_override @inline candidate_object_ray() =
    (Vec3f(Metal.candidate_ox(), Metal.candidate_oy(), Metal.candidate_oz()),
     Vec3f(Metal.candidate_dx(), Metal.candidate_dy(), Metal.candidate_dz()))

# A no-op: a payload calls this only for a `t` that improves its running best and
# returns that best, so the MSL loop compares the returned `t` with the one it
# sent and issues `commit_bounding_box_intersection` itself.
@device_override @inline commit_intersection!(t) = nothing
