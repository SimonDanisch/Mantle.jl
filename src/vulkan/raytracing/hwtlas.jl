# Vulkan's half of core's `BatchedTLAS` (`raytracing/mutation.jl`): the native verbs a build
# and a trace need, the query verbs on Lava's inline ray query, and the descriptor binding.

import Raycore
import Adapt
using GeometryBasics
using StaticArrays: SVector, SMatrix
using Base: @propagate_inbounds
import KernelAbstractions
const KA = KernelAbstractions

const Mat4f = SMatrix{4, 4, Float32, 16}

blastype(::LavaDevice) = LavaBLAS

# Read by the build as its input: `AS_INPUT_USAGE`.
nativerecords(d::LavaDevice, n::Integer) =
    LavaArray{VulkanInstanceRecord, 1}(undef, Int(n); bq = batchqueue(d), extra_usage = AS_INPUT_USAGE)

# An instance names its BLAS by device address.
geometryvalue(blas::LavaBLAS, ::Integer) = blas.address

# Bound as a descriptor (`bindtlas!`), not passed as a value.
scenevalue(::LavaTLAS) = nothing

"""Drop the host owner and adapt traversal arrays; keep the out-of-band scene value.
Descriptor discovery uses the original unadapted hwtlas reference."""
Adapt.adapt_structure(to::LavaAdaptor, accel::AdaptedAccel) =
    adaptaccel(to, accel)

# No shader binding table, so the offset is 0; the flags force the instance opaque or not
# (FORCE_OPAQUE 0x4, FORCE_NO_OPAQUE 0x8), whatever its geometry was built as.
@inline instancerecord(::Type{VulkanInstanceRecord}, transform, id::UInt32, mask::UInt32,
                       blas_address::UInt64, opaque::Bool) =
    VulkanInstanceRecord(transform, id | (mask << 24), (opaque ? UInt32(0x4) : UInt32(0x8)) << 24,
                         blas_address)

# The query verbs (`raytracing/accel.jl`) on Lava's inline ray query. A kernel
# tracing through an `AdaptedAccel` is compiled with `enable_ray_query=true` (set
# when `lava_launch!` gets `tlas=`), which binds the TLAS descriptor at set 0,
# binding 0. The query object lives in the shader, one per invocation, so the
# handle carries nothing.
struct VulkanRayQuery end

# SPIR-V ray-query intersection kinds. Named because `1` means AABB for a
# candidate and triangle for a committed hit.
const CANDIDATE_AABB     = UInt32(1)
const COMMITTED_TRIANGLE  = UInt32(1)
const COMMITTED_GENERATED = UInt32(2)

# `mask` is the ray's cull mask; the driver compares its low 8 bits with each
# instance's mask. `% UInt32` avoids an error branch in every shader. Ray flags:
# OpaqueKHR (1) takes every triangle as opaque, TerminateOnFirstHitKHR (4) stops at
# the first commit.
@inline function query_init(accel::AdaptedAccel{<:Any, <:LavaDeviceArray}, ray::Raycore.AbstractRay,
                            mask::Integer, flags::UInt32)
    o = ray.o; d = ray.d
    lava_ray_query_init(flags, mask % UInt32,
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return VulkanRayQuery()
end

@inline query_begin(accel::AdaptedAccel{<:Any, <:LavaDeviceArray}, ray::Raycore.AbstractRay,
                    mask::Integer, anyhit::Bool) =
    query_init(accel, ray, mask, anyhit ? UInt32(1 | 4) : UInt32(1))
@inline query_begin_transparent(accel::AdaptedAccel{<:Any, <:LavaDeviceArray}, ray::Raycore.AbstractRay,
                                mask::Integer) =
    query_init(accel, ray, mask, UInt32(4))

@inline query_next!(::VulkanRayQuery) = lava_ray_query_proceed()
@inline candidate_is_box(::VulkanRayQuery) = lava_ray_query_get_type(false) == CANDIDATE_AABB
@inline candidate_primitive_index(::VulkanRayQuery) = Int(lava_ray_query_get_primitive_index(false)) + 1
@inline candidate_ids(::VulkanRayQuery) =
    (lava_ray_query_get_instance_id(false), lava_ray_query_get_primitive_index(false),
     lava_ray_query_get_instance_custom_index(false))
@inline function candidate_triangle(::VulkanRayQuery)
    u, v = lava_ray_query_get_barycentrics(false)
    return (lava_ray_query_get_t(false), u, v)
end
@inline commit_triangle!(::VulkanRayQuery) = lava_ray_query_confirm()
@inline candidate_object_ray(::VulkanRayQuery) = (
    Vec3f(lava_ray_query_get_object_ray_origin(false, 1),
          lava_ray_query_get_object_ray_origin(false, 2),
          lava_ray_query_get_object_ray_origin(false, 3)),
    Vec3f(lava_ray_query_get_object_ray_direction(false, 1),
          lava_ray_query_get_object_ray_direction(false, 2),
          lava_ray_query_get_object_ray_direction(false, 3)))
@inline commit_intersection!(::VulkanRayQuery, t) = lava_ray_query_generate_intersection(Float32(t))

@inline function committed_kind(::VulkanRayQuery)
    kind = lava_ray_query_get_type(true)
    kind == COMMITTED_TRIANGLE && return Raycore.HIT_TRIANGLE
    kind == COMMITTED_GENERATED && return Raycore.HIT_GENERATED
    return Raycore.HIT_MISS
end
@inline committed_ids(::VulkanRayQuery) =
    (lava_ray_query_get_t(true), lava_ray_query_get_instance_id(true),
     lava_ray_query_get_primitive_index(true), lava_ray_query_get_instance_custom_index(true))
@inline committed_barycentrics(::VulkanRayQuery) = lava_ray_query_get_barycentrics(true)
@inline query_end!(::VulkanRayQuery) = nothing

"""
    tlasset!(owner, ctx, pipeline, tlas) -> VkDescriptorSet

The descriptor set binding `tlas` for `pipeline`'s layout, owned by the owner of
the commands that bind it. A [`Recording`](@ref), whose commands bake the set,
keeps one per (layout, structure) pair for as long as it can be submitted; a
one-shot allocates one per dispatch and holds its pool. Not a keyed cache:
`Raycore.sync!` makes a new `LavaTLAS` per rebuild, so one would grow without
bound or evict pools still in flight. Only `release!` frees.
"""
function tlasset!(rec::Recording, ctx::LavaDevice, pipeline::LavaComputePipeline, tlas)
    layout = pipeline.descriptor_set_layout::VK.DescriptorSetLayout
    for (l, t, _, set) in rec.sets
        (l === layout && t === tlas) && return set
    end
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(ctx, pipeline, tlas)
    push!(rec.sets, (layout, tlas, desc_pool, desc_set))
    return desc_set
end

function tlasset!(o::OneShot, ctx::LavaDevice, pipeline::LavaComputePipeline, tlas)
    desc_pool, desc_set = alloc_compute_tlas_descriptor_set(ctx, pipeline, tlas)
    # The pool must outlive the submission that binds the set.
    hold!(o, desc_pool)
    return desc_set
end

function bindtlas!(e::Emitter, pipeline::LavaComputePipeline, tlas::BatchedTLAS{<:Any,LavaDevice})
    lava_tlas = tlas.built::LavaTLAS
    set = tlasset!(e.owner, e.ctx, pipeline, lava_tlas)
    VK.cmd_bind_descriptor_sets(e.cmd, VK.PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout, UInt32(0), [set], UInt32[])
    # One hold covers both levels: the `LavaTLAS` references every `LavaBLAS`,
    # and `syncbuf!(::LavaTLAS)` records each level's storage.
    hold!(e, lava_tlas)
    return nothing
end

# The inline ray query answers boxes through the candidate verbs above.
supports_procedural_traversal(d::LavaDevice) = d.ray_query_available
