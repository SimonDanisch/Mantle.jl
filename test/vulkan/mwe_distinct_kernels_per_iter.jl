# MWE-4: many distinct kernel types per iter (matches Hikari's
# LINKED_KERNEL_CACHE growth pattern of ~12-15 unique entries).
#
# Hikari has many distinct kernel functions (gpu_workqueue_map_kernel
# with different `f` parameters, gpu_extract_rays_kernel, gpu_init_*,
# etc.), each compiled separately and cached in LINKED_KERNEL_CACHE.
# By iter 6, Lava has executed ~600 distinct (kernel, args)
# combinations.  Maybe the cascade fault needs cache-pressure-induced
# pipeline reuse with stale state.

using Raycore, Lava, GeometryBasics, StaticArrays, LinearAlgebra
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx = MVE.vk_context()

hwtlas = MVE.VulkanTLAS(backend)
mesh = GeometryBasics.normal_mesh(GeometryBasics.Tessellation(
    GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 8))
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
Raycore.sync!(hwtlas)

# 12 distinct kernels — each is a unique closure → unique cache entry.
@kernel function _k_a!(out, val); i = @index(Global); out[i] = val + 1f0; end
@kernel function _k_b!(out, val); i = @index(Global); out[i] = val + 2f0; end
@kernel function _k_c!(out, val); i = @index(Global); out[i] = val + 3f0; end
@kernel function _k_d!(out, val); i = @index(Global); out[i] = val + 4f0; end
@kernel function _k_e!(out, val); i = @index(Global); out[i] = val + 5f0; end
@kernel function _k_f!(out, val); i = @index(Global); out[i] = val + 6f0; end
@kernel function _k_g!(out, val); i = @index(Global); out[i] = val + 7f0; end
@kernel function _k_h!(out, val); i = @index(Global); out[i] = val + 8f0; end
@kernel function _k_i!(out, val); i = @index(Global); out[i] = val + 9f0; end
@kernel function _k_j!(out, val); i = @index(Global); out[i] = val + 10f0; end
@kernel function _k_k!(out, val); i = @index(Global); out[i] = val + 11f0; end
@kernel function _k_l!(out, val); i = @index(Global); out[i] = val + 12f0; end
const KERNS = (_k_a!, _k_b!, _k_c!, _k_d!, _k_e!, _k_f!, _k_g!, _k_h!, _k_i!, _k_j!, _k_k!, _k_l!)

println("=== MWE-4: 12 distinct kernels + indirect RT + per-iter alloc/free ===")
const N_ITERS = 20
crashed_at = 0
for iter in 1:N_ITERS
    bufs = [MVE.LavaArray(zeros(Float32, 1024)) for _ in 1:12]
    rays = MVE.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:1024])
    hits = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), 1024))
    n_buf = MVE.LavaArray(Int32[1024])

    for sample in 1:4
        for (k, kern) in enumerate(KERNS)
            kern(backend)(bufs[k], Float32(sample); ndrange=1024)
        end
        Mantle.trace_closest_hits_indirect!(hits, rays, hwtlas.hw_accel, n_buf)
        for (k, kern) in enumerate(KERNS)
            kern(backend)(bufs[k], Float32(sample+1); ndrange=1024)
        end
        KA.synchronize(backend)
    end

    bufs = nothing; rays = nothing; hits = nothing; n_buf = nothing

    if MVE.device_lost(ctx)
        global crashed_at = iter
        break
    end
    print("$iter ")
end
println()
println(crashed_at == 0 ? "MWE-4: $N_ITERS iters clean." : "MWE-4: !!! crashed at iter $crashed_at")

using Test
@test crashed_at == 0
@test !MVE.device_lost(ctx)
