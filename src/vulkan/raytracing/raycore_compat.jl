# Raycore-compatible hardware tracing: `HardwareAccel` and `trace_closest_hits!`
# stand in for Raycore's software `closest_hit`.

# Raycore's ray and hit layouts, shared with the shaders below.
using Raycore: RTRay, RTHitResult

"""HardwareAccel{TriVec}

Raycore import and tracing context: native TLAS, host primitive lookup tables,
portable shader templates and the submit channel traces run on.
"""
mutable struct HardwareAccel{TriVec <: AbstractVector}
    tlas::LavaTLAS
    triangle_data::TriVec           # host primitives for lookup
    blas_offsets::Vector{UInt32}
    # `per_instance_tri_offsets[iid+1]`: where instance `iid`'s BLAS triangles
    # start in `triangle_data`, so `custom_index` stays free for the caller.
    per_instance_tri_offsets::Vector{UInt32}
    rt_pipeline::RayTracingPipeline
    # Set by `set_anyhit_pipeline!`.
    anyhit_pipeline::Union{Nothing, RayTracingPipeline}
    # Channel this accel's traces run on.
    bq::SubmitChannel{<:VulkanQueue}
end

"""HardwareAccel(tlas; ctx, bq = ctx.default_bq)

Import a Raycore.TLAS or compatible owning BLAS/instance source into pooled
native storage on bq. Canonical row-major transforms, custom IDs, masks and
visibility are preserved. Raycore.TLAS uses its owning blas_storage arrays.
"""
function HardwareAccel(tlas;
                       ctx::VkContext,
                       bq::SubmitChannel{<:VulkanQueue}=ctx.default_bq)
    hw_tlas, tri_data, offsets, per_inst_offsets = build_hw_accel_from_tlas(tlas; ctx, bq)
    HardwareAccel(hw_tlas, tri_data, offsets, per_inst_offsets; bq)
end

"""HardwareAccel(hw_tlas, triangle_data, blas_offsets, per_instance_tri_offsets; bq)

Wrap prebuilt native storage and CPU lookup tables. The default channel belongs
to hw_tlas. Creates portable RT shader templates; native programs are prepared
on the tracing device. Mesh-swap callers may retain these templates while
replacing geometry and offset fields.
"""
function HardwareAccel(hw_tlas::LavaTLAS, triangle_data::TriVec, blas_offsets,
                       per_instance_tri_offsets::AbstractVector{UInt32};
                       bq::SubmitChannel{<:VulkanQueue}=(hw_tlas.storage.buf[].ctx::VkContext).default_bq
                       ) where {TriVec <: AbstractVector}
    pipeline = RayTracingPipeline(
        raygen=hw_raygen,
        closest_hit=hw_closesthit,
        miss=hw_miss,
        payload_type=:f32_7,
    )
    HardwareAccel{TriVec}(hw_tlas, triangle_data, blas_offsets, collect(per_instance_tri_offsets),
                         pipeline, nothing, bq)
end

"""
    set_anyhit_pipeline!(accel::HardwareAccel, anyhit_func, raygen_func)

Create and cache an any-hit pipeline variant for this HardwareAccel.
The `anyhit_func` and `raygen_func` must have matching BDA arg signatures.
"""
function set_anyhit_pipeline!(accel::HardwareAccel, anyhit_func, raygen_func)
    accel.anyhit_pipeline = RayTracingPipeline(
        raygen=raygen_func,
        closest_hit=hw_closesthit,
        miss=hw_miss,
        any_hit=anyhit_func,
        payload_type=:f32_7,
    )
    return accel
end

"""
    trace_closest_hits!(results, rays, accel::HardwareAccel, n_rays::Integer;
                        cull_mask::UInt32 = UInt32(0xFF))

Trace `n_rays` rays and write one `RTHitResult` each into `results`
(`LavaArray{RTRay}` / `LavaArray{RTHitResult}`, or anything `trace_rays!` accepts).
An instance is visible to a ray only when `cull_mask & instance_mask != 0`.
"""
function trace_closest_hits!(results, rays, accel::HardwareAccel, n_rays::Integer;
                              cull_mask::UInt32 = UInt32(0xFF))
    trace_rays!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results, cull_mask;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32};
                                  cull_mask::UInt32 = UInt32(0xFF))

[`trace_closest_hits!`](@ref) with the ray count read from `n_rays` on the device.
"""
function trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32};
                                       cull_mask::UInt32 = UInt32(0xFF))
    trace_rays_indirect!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results, cull_mask; n_rays=n_rays)
end

"""
    trace_closest_hits_anyhit!(results, rays, accel, n_rays, extra_args...)

Trace with the pipeline from `set_anyhit_pipeline!`; `extra_args` follow
`(rays, results)` in the raygen and any-hit arguments. There is no `cull_mask`:
the any-hit raygen functions pick their own.
"""
function trace_closest_hits_anyhit!(results, rays, accel::HardwareAccel, n_rays::Integer, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_anyhit_indirect!(results, rays, accel, n_rays_buf, extra_args...)

[`trace_closest_hits_anyhit!`](@ref) with the ray count read on the device.
"""
function trace_closest_hits_anyhit_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32}, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays_indirect!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...; n_rays=n_rays)
end

# Built-in shaders. Payload slots: hit, t, primitive id, custom index, u, v, instance id.

function hw_raygen(rays::LavaDeviceArray{RTRay,1},
                   results::LavaDeviceArray{RTHitResult,1},
                   cull_mask::UInt32)
    lid = rt_launch_id_x()

    ray = rays[lid + 1]

    # Initialize payload to miss
    lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    lava_rt_payload_store_f32_at(-1f0, UInt32(1))  # t=-1

    lava_rt_trace_ray(
        UInt32(0),    # flags
        cull_mask,    # cull mask
        UInt32(0),    # sbt offset
        UInt32(0),    # sbt stride
        UInt32(0),    # miss index
        ray.origin_x, ray.origin_y, ray.origin_z, ray.tmin,
        ray.dir_x, ray.dir_y, ray.dir_z, ray.tmax
    )

    hit  = lava_rt_payload_load_f32_at(UInt32(0))
    t    = lava_rt_payload_load_f32_at(UInt32(1))
    pid  = lava_rt_payload_load_f32_at(UInt32(2))
    ci   = lava_rt_payload_load_f32_at(UInt32(3))
    bu   = lava_rt_payload_load_f32_at(UInt32(4))
    bv   = lava_rt_payload_load_f32_at(UInt32(5))
    iid  = lava_rt_payload_load_f32_at(UInt32(6))

    results[lid + 1] = RTHitResult(
        reinterpret(UInt32, hit),
        t,
        reinterpret(UInt32, pid),
        reinterpret(UInt32, ci),
        bu, bv,
        reinterpret(UInt32, iid),
        UInt32(0),
    )
    return nothing
end

function hw_closesthit()
    t = rt_ray_tmax()
    ci = rt_instance_custom_index()
    iid = rt_instance_id()
    pid = rt_primitive_id()
    bu = rt_hit_bary_u()
    bv = rt_hit_bary_v()

    lava_rt_payload_store_f32_at(reinterpret(Float32, UInt32(1)), UInt32(0))  # hit=1
    lava_rt_payload_store_f32_at(t, UInt32(1))
    lava_rt_payload_store_f32_at(reinterpret(Float32, pid), UInt32(2))
    lava_rt_payload_store_f32_at(reinterpret(Float32, ci), UInt32(3))
    lava_rt_payload_store_f32_at(bu, UInt32(4))
    lava_rt_payload_store_f32_at(bv, UInt32(5))
    lava_rt_payload_store_f32_at(reinterpret(Float32, iid), UInt32(6))
    return nothing
end

function hw_miss()
    lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    lava_rt_payload_store_f32_at(-1f0, UInt32(1))   # t=-1
    lava_rt_payload_store_f32_at(0f0, UInt32(2))
    lava_rt_payload_store_f32_at(0f0, UInt32(3))
    lava_rt_payload_store_f32_at(0f0, UInt32(4))
    lava_rt_payload_store_f32_at(0f0, UInt32(5))
    lava_rt_payload_store_f32_at(0f0, UInt32(6))    # instance_id irrelevant on miss
    return nothing
end
