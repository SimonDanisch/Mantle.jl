# Hardware ray traversal on Metal.
#
# ── Why this file is MSL source and not a Julia kernel ───────────────────────
#
# Every other kernel this backend runs is Julia, compiled to AIR by Metal.jl.
# Traversal cannot be, and the reason is structural rather than a gap someone
# can fill in Metal.jl: `metal::raytracing::intersector<>` is a C++ class
# template whose traversal the Metal *frontend* instantiates and inlines. It is
# not an `extern` AIR function, so there is no symbol for `@typed_ccall` to
# name and no `@device_override` that could become one. Confirmed by looking:
# no `air.intersect*` appears anywhere in Metal.jl or in AIR's intrinsic set.
#
# What IS available is `newLibraryWithSource:`, which runs that frontend at
# runtime with no Xcode installed. So the traversal is written once, in MSL,
# compiled on first use and cached per device.
#
# This does not put shading in MSL. `trace_closest_hits!` is the ray-query
# contract — rays in, nearest hit out, no shader callback — which is exactly
# what a self-contained kernel can express. Hikari's wavefront integrator reads
# the hit buffer back in Julia and shades there, so the Julia/MSL boundary sits
# where the data does.
#
# ── The layouts are Raycore's, byte for byte ─────────────────────────────────
#
# `RTRay` and `RTHitResult` are 32-byte flat structs of Float32/UInt32 with no
# padding, which is why they can be mirrored in MSL at all. The asserts below
# check that rather than trusting it: a field added on the Julia side would
# otherwise silently shear every hit record.

const TRACE_MSL = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct RTRay {
    float origin_x, origin_y, origin_z, tmin;
    float dir_x, dir_y, dir_z, tmax;
};

struct RTHitResult {
    uint  hit;
    float t;
    uint  primitive_id;
    uint  instance_custom_index;
    float bary_u, bary_v;
    uint  instance_id;
    uint  pad;
};

kernel void trace_closest_hits(
    instance_acceleration_structure accel [[buffer(0)]],
    device const RTRay  *rays  [[buffer(1)]],
    device RTHitResult  *hits  [[buffer(2)]],
    constant uint       &n     [[buffer(3)]],
    constant uint       &cull  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n) return;

    RTRay q = rays[tid];
    ray r;
    r.origin       = float3(q.origin_x, q.origin_y, q.origin_z);
    r.direction    = float3(q.dir_x, q.dir_y, q.dir_z);
    r.min_distance = q.tmin;
    r.max_distance = q.tmax;

    intersector<instancing, triangle_data> isect;
    intersection_result<instancing, triangle_data> res = isect.intersect(r, accel, cull);

    RTHitResult h;
    h.pad = 0u;
    if (res.type == intersection_type::none) {
        // A miss reports tmax, matching the software traversal: callers test
        // `hit`, but anything that reads `t` regardless must not see a stale
        // or uninitialised distance.
        h.hit = 0u; h.t = q.tmax;
        h.primitive_id = 0u; h.instance_custom_index = 0u;
        h.bary_u = 0.0f; h.bary_v = 0.0f; h.instance_id = 0u;
    } else {
        h.hit = 1u;
        h.t = res.distance;
        h.primitive_id = res.primitive_id;
        h.instance_id  = res.instance_id;
        // Metal's plain instance descriptor carries no user ID, so the custom
        // index is the instance index. A scene that needs Vulkan's separate
        // `instanceCustomIndex` has to build with
        // MTLAccelerationStructureUserIDInstanceDescriptor and read
        // `user_instance_id` here.
        h.instance_custom_index = res.instance_id;
        float2 bc = res.triangle_barycentric_coord;
        h.bary_u = bc.x;
        h.bary_v = bc.y;
    }
    hits[tid] = h;
}
"""

# The Julia side of the ABI above. Checked at load, not assumed.
@assert sizeof(Raycore.RTRay) == 32       "RTRay must be 32 bytes to match TRACE_MSL; got $(sizeof(Raycore.RTRay))"
@assert sizeof(Raycore.RTHitResult) == 32 "RTHitResult must be 32 bytes to match TRACE_MSL; got $(sizeof(Raycore.RTHitResult))"
@assert fieldnames(Raycore.RTRay) ==
        (:origin_x, :origin_y, :origin_z, :tmin, :dir_x, :dir_y, :dir_z, :tmax)
@assert fieldnames(Raycore.RTHitResult) ==
        (:hit, :t, :primitive_id, :instance_custom_index, :bary_u, :bary_v, :instance_id, :_pad2)

"""
Compile the traversal kernel once per device and keep it.

`newLibraryWithSource:` runs the Metal frontend, which costs tens of
milliseconds — fine once, not per trace.
"""
function trace_pipeline(d::MetalDevice)
    p = d.trace_pipeline
    p === nothing || return p
    lib = MTL.MTLLibrary(d.dev, TRACE_MSL)
    fun = MTL.MTLFunction(lib, "trace_closest_hits")
    return d.trace_pipeline = MTL.MTLComputePipelineState(d.dev, fun)
end

# The MTLBuffer behind a device array, and the byte offset into it. Mantle's
# `deviceview` hands out borrowed `MtlArray`s over pool regions, so the offset
# is load-bearing — binding the buffer at 0 would read another tenant's bytes.
_mtlbuffer(a::MtlArray) = (a.data[], a.offset * sizeof(eltype(a)))

"""
    trace_closest_hits!(hits, rays, tlas, n_rays; cull_mask = 0xFF)

Trace `n_rays` from `rays` against `tlas`, writing the nearest hit of each into
`hits`. Hardware traversal.

`cull_mask` is ANDed with each instance's mask, as in Vulkan.
"""
function trace_closest_hits!(hits::MtlArray, rays::MtlArray, tlas::MetalTLAS,
                             n_rays::Integer; cull_mask::UInt32 = UInt32(0xFF))
    n_rays == 0 && return hits
    eltype(rays) === Raycore.RTRay ||
        throw(ArgumentError("trace_closest_hits!: rays must be RTRay, got $(eltype(rays))"))
    eltype(hits) === Raycore.RTHitResult ||
        throw(ArgumentError("trace_closest_hits!: hits must be RTHitResult, got $(eltype(hits))"))
    length(rays) >= n_rays ||
        throw(ArgumentError("trace_closest_hits!: rays holds $(length(rays)) < n_rays=$(n_rays)"))
    length(hits) >= n_rays ||
        throw(ArgumentError("trace_closest_hits!: hits holds $(length(hits)) < n_rays=$(n_rays)"))

    d = Device(MetalAPI())
    pip = trace_pipeline(d)
    rbuf, roff = _mtlbuffer(rays)
    hbuf, hoff = _mtlbuffer(hits)
    n = UInt32(n_rays)
    cm = cull_mask

    cmdbuf = MTL.MTLCommandBuffer(d.queue)
    MTL.MTLComputeCommandEncoder(cmdbuf) do cce
        MTL.set_function!(cce, pip)
        MTL.set_acceleration_structure!(cce, tlas.handle, 1)
        # Residency for the structures the TLAS points at. Binding the TLAS
        # alone leaves its BLASes non-resident and traversal silently misses.
        isempty(tlas.blases) || MTL.use!(cce, tlas.blases, MTL.ReadUsage)
        MTL.set_buffer!(cce, rbuf, roff, 2)
        MTL.set_buffer!(cce, hbuf, hoff, 3)
        MTL.set_bytes!(cce, Base.unsafe_convert(Ptr{Cvoid}, Ref(n)), sizeof(UInt32), 4)
        MTL.set_bytes!(cce, Base.unsafe_convert(Ptr{Cvoid}, Ref(cm)), sizeof(UInt32), 5)
        w = min(Int(pip.maxTotalThreadsPerThreadgroup), 256)
        MTL.dispatchThreads!(cce, MTL.MTLSize(n_rays, 1, 1), MTL.MTLSize(w, 1, 1))
    end
    MTL.commit!(cmdbuf)
    MTL.wait_completed(cmdbuf)
    return hits
end
