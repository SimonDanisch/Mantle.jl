# MWE-3: indirect RT + many interleaved kernels per iter (mimicks Hikari's
# wavefront pattern more closely).
#
# Hikari's pattern per render:
#   ~12 LavaArrays alloc (VolPathState work queues + accumulators)
#   per sample (~4 samples):
#     dispatches: gen_camera_rays → workqueue_map → rt_indirect → many shadow/material kernels
#     ~23 batch submits per sample
#   total ~92 batch submits per render

using Raycore, Lava, GeometryBasics, StaticArrays, LinearAlgebra
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx = Mantle.vk_context()

hwtlas = Mantle.VulkanTLAS(backend)
mesh = GeometryBasics.normal_mesh(GeometryBasics.Tessellation(
    GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 8))
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
Raycore.sync!(hwtlas)

@kernel function _busy_kernel!(out, @Const(input), val)
    i = @index(Global)
    out[i] = input[i] + val
end

println("=== MWE-3: indirect RT + busy compute interleave + per-iter alloc/free ===")
const N_ITERS = 20
crashed_at = 0
for iter in 1:N_ITERS
    # 12 fresh LavaArrays per iter (mirrors VolPathState count).
    bufs = [Mantle.LavaArray(zeros(Float32, 1024)) for _ in 1:12]
    rays = Mantle.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:1024])
    hits = Mantle.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), 1024))
    n_buf = Mantle.LavaArray(Int32[1024])

    # Mimic Hikari's per-sample pattern: ~10 compute dispatches + indirect RT.
    for sample in 1:4
        for k in 1:5
            _busy_kernel!(backend)(bufs[mod1(k, 12)], bufs[mod1(k+1, 12)], Float32(sample); ndrange=1024)
        end
        Mantle.trace_closest_hits_indirect!(hits, rays, hwtlas.hw_accel, n_buf)
        for k in 6:10
            _busy_kernel!(backend)(bufs[mod1(k, 12)], bufs[mod1(k+1, 12)], Float32(sample); ndrange=1024)
        end
        KA.synchronize(backend)
    end

    bufs = nothing; rays = nothing; hits = nothing; n_buf = nothing

    if Mantle.device_lost(ctx)
        global crashed_at = iter
        break
    end
end
if crashed_at == 0
    println("MWE-3: $N_ITERS iters clean.")
else
    println("MWE-3: !!! crashed at iter $crashed_at — found minimal trigger!")
end

using Test
@test crashed_at == 0
@test !Mantle.device_lost(ctx)
