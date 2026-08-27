# RayCore-compatible hardware ray tracing interface for Lava.jl
#
# Provides HardwareAccel and trace_closest_hits!() as a drop-in replacement
# for Raycore's software BVH traversal closest_hit().

# Use Raycore's RTRay and RTHitResult (same memory layout, single definition)
using Raycore: RTRay, RTHitResult

"""
    HardwareAccel{TriVec <: AbstractVector}

Hardware-accelerated ray tracing context. Built from a Raycore-compatible TLAS.
Parametrised on the concrete triangle-vector type (e.g.
`Vector{Raycore.Triangle{UInt32}}`) so field accesses stay type-stable.

# Fields
- `tlas::LavaTLAS` — Vulkan top-level acceleration structure
- `triangle_data::TriVec` — CPU vector of all primitives (for lookup after trace)
- `blas_offsets::Vector{UInt32}` — Per-BLAS offset into triangle_data
- `rt_pipeline::RayTracingPipeline` — Pre-compiled raygen+closesthit+miss

# Usage
```julia
hw = HardwareAccel(raycore_tlas)
results = LavaArray{RTHitResult}(n_rays)
trace_closest_hits!(results, rays, hw)
```
"""
mutable struct HardwareAccel{TriVec <: AbstractVector}
    tlas::LavaTLAS
    triangle_data::TriVec           # CPU primitives for lookup
    blas_offsets::Vector{UInt32}
    # Per-instance triangle-array offset, indexed by `gl_InstanceID`.
    # `per_instance_tri_offsets[iid+1]` = offset into `triangle_data` where
    # that instance's BLAS triangles start.  This removes the old
    # `off_gpu[custom_index + 1]` lookup, freeing `custom_index` to carry
    # the interface override instead of the BLAS index.
    per_instance_tri_offsets::Vector{UInt32}
    rt_pipeline::RayTracingPipeline
    # Optional any-hit pipeline (lazy — created on first use via set_anyhit_pipeline!)
    anyhit_pipeline::Union{Nothing, RayTracingPipeline}
    # BatchQueue this accel's RT dispatches run on.  Stored explicitly so
    # callers don't reach for an implicit `vk_context().default_bq`.
    bq::BatchQueue
end

"""
    HardwareAccel(tlas; bq=<derived from tlas storage>) -> HardwareAccel

Build a HardwareAccel from a Raycore-compatible TLAS.
The TLAS must have `.blas_array` and `.instances` fields.

`bq` defaults to the BatchQueue whose ctx built the TLAS (the CPU-side TLAS
object is used to find the ctx via the HW build path), so ray tracing runs
on the same device the AS was allocated against.
"""
function HardwareAccel(tlas;
                       ctx::VkContext=vk_context(),
                       bq::BatchQueue=ctx.default_bq)
    hw_tlas, tri_data, offsets, per_inst_offsets = build_hw_accel_from_tlas(tlas; ctx)
    HardwareAccel(hw_tlas, tri_data, offsets, per_inst_offsets; bq)
end

"""
    HardwareAccel(hw_tlas::LavaTLAS, triangle_data, blas_offsets, per_instance_tri_offsets;
                  bq=<derived from hw_tlas>) -> HardwareAccel

Build a HardwareAccel from a pre-built Vulkan TLAS + triangle data + offsets.
Default `bq` is taken from `hw_tlas.storage.buf[].ctx.default_bq` so we never
mismatch the AS against a queue on a different device.

A fresh `RayTracingPipeline` (and its SBT buffer) is built per HardwareAccel.
Callers that want to avoid the per-frame SBT allocation during mesh-swap
rebuilds should reuse the previous HardwareAccel via the Raycore-side
`build_hw_tlas(...; accel_prev=hwtlas.hw_accel)` plumbing: that updates the
geometry-dependent fields (`tlas`, `triangle_data`, `blas_offsets`,
`per_instance_tri_offsets`) in place and keeps the pipeline alive. In the
Raycore/RayMakie flow this happens automatically on every `sync!(hwtlas)` —
one pipeline per HWTLAS, lifetime tied to the HWTLAS.
"""
function HardwareAccel(hw_tlas::LavaTLAS, triangle_data::TriVec, blas_offsets,
                       per_instance_tri_offsets::AbstractVector{UInt32};
                       bq::BatchQueue=(hw_tlas.storage.buf[].ctx::VkContext).default_bq
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

Trace `n_rays` rays against the hardware acceleration structure.
Results are written to `results` buffer (one RTHitResult per ray).

`results` and `rays` can be `LavaArray{RTHitResult}`/`LavaArray{RTRay}`,
or any type accepted by `trace_rays!`.

`cull_mask` is ANDed against each instance's instance_mask in the TLAS.
An instance is visible to a ray only when `(cull_mask & instance_mask) != 0`.
Default `0xFF` matches all instances (backward-compatible).
"""
function trace_closest_hits!(results, rays, accel::HardwareAccel, n_rays::Integer;
                              cull_mask::UInt32 = UInt32(0xFF))
    trace_rays!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results, cull_mask;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32};
                                  cull_mask::UInt32 = UInt32(0xFF))

Indirect RT trace -- reads ray count from GPU buffer. No CPU readback.

`cull_mask` is ANDed against each instance's instance_mask in the TLAS.
Default `0xFF` matches all instances (backward-compatible).
"""
function trace_closest_hits_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32};
                                       cull_mask::UInt32 = UInt32(0xFF))
    trace_rays_indirect!(accel.bq, accel.rt_pipeline, accel.tlas, rays, results, cull_mask; n_rays=n_rays)
end

"""
    trace_closest_hits_anyhit!(results, rays, accel, n_rays, extra_args...)

RT trace with any-hit shader. `extra_args` are passed after (rays, results) to the
raygen and any-hit functions via the shared BDA arg buffer.

TODO(P3-followup): cull_mask plumbing for the anyhit variants is deferred.
The anyhit raygen functions have separate signatures (extra_args at position 3+);
adding cull_mask requires coordinated changes to those raygen signatures.
For now cull_mask is hardcoded 0xFF inside the respective raygen functions.
"""
function trace_closest_hits_anyhit!(results, rays, accel::HardwareAccel, n_rays::Integer, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...;
                width=Int(n_rays), height=1)
end

"""
    trace_closest_hits_anyhit_indirect!(results, rays, accel, n_rays_buf, extra_args...)

Indirect RT trace with any-hit shader -- reads ray count from GPU buffer.

TODO(P3-followup): cull_mask plumbing deferred -- see trace_closest_hits_anyhit!.
"""
function trace_closest_hits_anyhit_indirect!(results, rays, accel::HardwareAccel, n_rays::LavaArray{Int32}, extra_args...)
    pipeline = accel.anyhit_pipeline
    pipeline === nothing && error("No any-hit pipeline set. Call set_anyhit_pipeline! first.")
    trace_rays_indirect!(accel.bq, pipeline, accel.tlas, rays, results, extra_args...; n_rays=n_rays)
end

# ── Built-in RT Shaders ──

function hw_raygen(rays::LavaDeviceArray{RTRay,1},
                   results::LavaDeviceArray{RTHitResult,1},
                   cull_mask::UInt32)
    lid = lava_rt_launch_id_x()

    ray = rays[lid + 1]

    # Initialize payload to miss
    lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    lava_rt_payload_store_f32_at(-1f0, UInt32(1))  # t=-1

    lava_rt_trace_ray(
        UInt32(0),    # flags
        cull_mask,    # cull mask (caller-supplied; default 0xFF matches all instances)
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
    t = lava_rt_ray_tmax()
    ci = lava_rt_instance_custom_index()
    iid = lava_rt_instance_id()
    pid = lava_rt_primitive_id()
    bu = lava_rt_hit_bary_u()
    bv = lava_rt_hit_bary_v()

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
