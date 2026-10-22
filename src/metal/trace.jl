# Hardware ray traversal on Metal, written in MSL: `intersector<>` is a C++
# template the Metal frontend instantiates, with no AIR intrinsic Metal.jl could
# call. `newLibraryWithSource:` compiles it at runtime without Xcode. Shading
# stays in Julia; the kernel only writes the nearest hit per ray.
#
# `RTRay` and `RTHitResult` mirror Raycore's 32-byte, padding-free layouts. The
# asserts below catch a Julia-side field change that would shear every hit record.

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

@assert sizeof(Raycore.RTRay) == 32       "RTRay must be 32 bytes to match TRACE_MSL; got $(sizeof(Raycore.RTRay))"
@assert sizeof(Raycore.RTHitResult) == 32 "RTHitResult must be 32 bytes to match TRACE_MSL; got $(sizeof(Raycore.RTHitResult))"
@assert fieldnames(Raycore.RTRay) ==
        (:origin_x, :origin_y, :origin_z, :tmin, :dir_x, :dir_y, :dir_z, :tmax)
@assert fieldnames(Raycore.RTHitResult) ==
        (:hit, :t, :primitive_id, :instance_custom_index, :bary_u, :bary_v, :instance_id, :_pad2)

"""
    trace_pipeline(dev) -> MTLComputePipelineState

The traversal pipeline, compiled on first use and cached on `dev`: compiling
from source costs tens of milliseconds.
"""
function trace_pipeline(d::MetalDevice)
    p = d.trace_pipeline
    p === nothing || return p
    lib = MTL.MTLLibrary(d.dev, TRACE_MSL)
    fun = MTL.MTLFunction(lib, "trace_closest_hits")
    return d.trace_pipeline = MTL.MTLComputePipelineState(d.dev, fun)
end

# The `MTLBuffer` behind an `MtlArray` and the byte offset into it. Views over
# pool regions share a buffer; binding at offset 0 reads another region's bytes.
mtlbuffer(a::MtlArray) = (a.data[].buffer, a.offset * sizeof(eltype(a)))

"""
    trace_closest_hits!(hits, rays, tlas, n_rays; cull_mask = 0xFF)

Trace `n_rays` from `rays` against `tlas` with hardware traversal, writing the
nearest hit of each into `hits`. `cull_mask` is ANDed with each instance's mask,
as in Vulkan.
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
    rbuf, roff = mtlbuffer(rays)
    hbuf, hoff = mtlbuffer(hits)
    n = UInt32(n_rays)
    cm = cull_mask

    cmdbuf = MTL.MTLCommandBuffer(cmdqueue(d))
    MTL.MTLComputeCommandEncoder(cmdbuf) do cce
        MTL.set_function!(cce, pip)
        MTL.set_acceleration_structure!(cce, tlas.handle, 1)
        # Binding the TLAS alone leaves its BLASes non-resident; traversal then
        # silently misses.
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
