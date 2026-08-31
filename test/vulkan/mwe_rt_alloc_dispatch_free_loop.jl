# MWE-2: HW RT version of MWE-1.
#
# Adds RT dispatch (VulkanTLAS + trace_closest_hits!) to the per-iter loop.
# Same alloc/free pattern as MWE-1, but with RT in the mix.
#
# If this crashes — the cascade is RT-specific (H3: vkCmdTraceRays
# referencing freed buffer; or H4 specific to RT-allocated BDAs).

using Raycore, Lava, GeometryBasics, StaticArrays, LinearAlgebra
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx = MVE.vk_context()

# One persistent VulkanTLAS shared across iters.
hwtlas = MVE.VulkanTLAS(backend)
mesh = GeometryBasics.normal_mesh(GeometryBasics.Tessellation(
    GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 8))
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
Raycore.sync!(hwtlas)

@kernel function _compute_kernel!(out, val)
    i = @index(Global)
    out[i] = val
end

println("=== MWE-2: HW RT + per-iter alloc/free ===")
const N_ITERS = 20
crashed_at = 0
for iter in 1:N_ITERS
    # Per-iter: allocate fresh buffers for ray + hits (mirrors Hikari pattern)
    rays = MVE.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:1024])
    hits = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), 1024))

    # Add some compute kernels too — like Hikari does.
    aux1 = MVE.LavaArray(zeros(Float32, 1024))
    aux2 = MVE.LavaArray(zeros(Float32, 1024))
    _compute_kernel!(backend)(aux1, Float32(iter); ndrange=1024)
    _compute_kernel!(backend)(aux2, Float32(iter); ndrange=1024)

    # RT dispatch
    Mantle.trace_closest_hits!(hits, rays, hwtlas.hw_accel, 1024)

    # More compute after RT
    _compute_kernel!(backend)(aux1, Float32(iter+1); ndrange=1024)

    KA.synchronize(backend)

    # Drop all per-iter buffers — finalizer thread frees them mid-loop.
    rays = nothing; hits = nothing; aux1 = nothing; aux2 = nothing

    if MVE.device_lost(ctx)
        global crashed_at = iter
        break
    end
end
if crashed_at == 0
    println("MWE-2: $N_ITERS iters clean — RT alloc/free loop is OK.")
else
    println("MWE-2: !!! crashed at iter $crashed_at — RT alloc/free triggers it!")
end

using Test
@test crashed_at == 0
@test !MVE.device_lost(ctx)
