# MWE-6: mirror Hikari VolPath's full per-render dispatch shape, per-iter alloc/free.
#
# Goal: get as close to Hikari's actual `render!` shape as possible while staying
# self-contained (no Hikari dep, no Raycore beyond VulkanTLAS).  Captures:
#   - 8 SoA "work queues" (StructArray of LavaArrays + Int32 size counter)
#   - per-bounce SoA pre-computed sample buffer
#   - pixel_L spectral accumulator + pixel_rgb + pixel_weight_sum
#   - per-sample: generate_camera_rays → for d in 1:max_depth (12+ kernels +
#     indirect HW RT for primary + 1-3 indirect HW RT for shadow rounds) →
#     accumulate_to_rgb → finalize_film
#   - per-iter alloc/free of all state, drop refs + GC.gc(false) (mirrors
#     `Base.close(vp::VolPath)`).
#
# If THIS reproduces the cascade fault → bug is in the alloc/free interaction
# with the dispatch shape itself, not in any Hikari-specific logic.
# If it does NOT → cascade is triggered by something specific to VolPath's
# kernel set or scene state we still need to capture.

using Raycore, Lava, GeometryBasics, StaticArrays, LinearAlgebra, StructArrays
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx = MVE.vk_context()

# Persistent VulkanTLAS shared across iters (mirrors Hikari: scene built once).
hwtlas = MVE.VulkanTLAS(backend)
mesh = GeometryBasics.normal_mesh(GeometryBasics.Tessellation(
    GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 8))
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
Raycore.sync!(hwtlas)

# === SoA work-item types (mirror VPRayWorkItem etc. shape) ===
struct RayWI;       a::Float32; b::Float32; c::Float32; flag::Bool;        end
struct MedSampWI;   a::Float32; b::Float32; flag::Bool;                    end
struct MedScatWI;   a::Float32; b::Float32; c::Float32;                    end
struct HitSurfWI;   a::Float32; b::Float32; flag::Bool;                    end
struct MatEvalWI;   a::Float32; b::Float32; c::Float32; d::Float32;        end
struct ShadowWI;    a::Float32; b::Float32; flag::Bool;                    end
struct EscapedWI;   a::Float32; b::Float32; c::Float32;                    end
struct RaySamples;  uc::Float32; u1::Float32; u2::Float32; rr::Float32; u3::Float32; end

# Mirrors should_use_soa(::Type{VPRayWorkItem}) = true.  Each SoA queue is a
# StructArray of LavaArrays + a tiny Int32 size buffer (atomic counter).
struct SoAQueue{T,V,S}
    items::V
    size::S
    capacity::Int32
end
function SoAQueue{T}(n::Int) where T
    fnames = fieldnames(T)
    ftypes = fieldtypes(T)
    cols = NamedTuple{fnames}(ntuple(i -> MVE.LavaArray(zeros(ftypes[i], n)), length(fnames)))
    items = StructArray{T}(cols)
    size  = MVE.LavaArray(Int32[0])
    SoAQueue{T,typeof(items),typeof(size)}(items, size, Int32(n))
end

@kernel function gen_camera_rays_kernel!(ray_q, ray_size, samples_q, val)
    i = @index(Global)
    ray_q[i]     = RayWI(val + 1f0, val + 2f0, val + 3f0, val > 0f0)
    samples_q[i] = RaySamples(val, val + 1f0, val + 2f0, val + 3f0, val + 4f0)
    ray_size[1]  = Int32(length(ray_q))
end
@kernel function trace_rays_compute_kernel!(ray_q, hit_q, esc_q, hit_size, esc_size, val)
    i = @index(Global)
    if i <= length(ray_q)
        hit_q[i] = HitSurfWI(val, val + 1f0, val > 0f0)
        esc_q[i] = EscapedWI(val, val + 1f0, val + 2f0)
        if i == 1
            hit_size[1] = Int32(length(ray_q))
            esc_size[1] = Int32(length(ray_q))
        end
    end
end
@kernel function medium_sample_kernel!(med_samp_q, med_scat_q, val)
    i = @index(Global)
    if i <= length(med_samp_q)
        med_scat_q[i] = MedScatWI(val, val + 1f0, val + 2f0)
    end
end
@kernel function medium_direct_lighting_kernel!(med_scat_q, shadow_q, val)
    i = @index(Global)
    if i <= length(med_scat_q)
        shadow_q[i] = ShadowWI(val, val + 1f0, val > 0f0)
    end
end
@kernel function medium_scatter_kernel!(med_scat_q, ray_q, samples_q, val)
    i = @index(Global)
    if i <= length(med_scat_q)
        s = samples_q[i]
        ray_q[i] = RayWI(s.uc + val, s.u1, s.u2, s.rr > 0f0)
    end
end
@kernel function escaped_kernel!(esc_q, pixel_L, val)
    i = @index(Global)
    if i <= length(esc_q)
        @inbounds pixel_L[i] += val
    end
end
@kernel function process_surface_hits_kernel!(hit_q, mat_q, pixel_L, val)
    i = @index(Global)
    if i <= length(hit_q)
        mat_q[i] = MatEvalWI(val, val + 1f0, val + 2f0, val + 3f0)
        @inbounds pixel_L[i] += val * 0.1f0
    end
end
@kernel function surface_direct_lighting_kernel!(mat_q, shadow_q, samples_q, val)
    i = @index(Global)
    if i <= length(mat_q)
        s = samples_q[i]
        shadow_q[i] = ShadowWI(s.uc + val, s.u1, s.rr > 0f0)
    end
end
@kernel function evaluate_materials_kernel!(mat_q, ray_q, samples_q, val)
    i = @index(Global)
    if i <= length(mat_q)
        s = samples_q[i]
        ray_q[i] = RayWI(s.uc + val, s.u1, s.u2, s.rr > 0f0)
    end
end
@kernel function accumulate_to_rgb_kernel!(pixel_L, pixel_rgb, weight_sum, val)
    i = @index(Global)
    L = pixel_L[i]
    @inbounds pixel_rgb[i]  += L * val
    @inbounds weight_sum[i] += val
end
@kernel function finalize_film_kernel!(pixel_rgb, weight_sum)
    i = @index(Global)
    w = weight_sum[i]
    if w > 0f0
        @inbounds pixel_rgb[i] /= w
    end
end

# === Per-render state alloc (mirrors VolPathState construction) ===
function alloc_state(n_pixels::Int)
    return (
        ray_a    = SoAQueue{RayWI}(n_pixels),
        ray_b    = SoAQueue{RayWI}(n_pixels),
        med_samp = SoAQueue{MedSampWI}(n_pixels),
        med_scat = SoAQueue{MedScatWI}(n_pixels),
        hit_surf = SoAQueue{HitSurfWI}(n_pixels),
        material = SoAQueue{MatEvalWI}(n_pixels),
        shadow   = SoAQueue{ShadowWI}(n_pixels),
        escaped  = SoAQueue{EscapedWI}(n_pixels),
        samples  = SoAQueue{RaySamples}(n_pixels).items,  # only items, no counter
        pixel_L     = MVE.LavaArray(zeros(Float32, n_pixels)),
        pixel_rgb   = MVE.LavaArray(zeros(Float32, n_pixels)),
        weight_sum  = MVE.LavaArray(zeros(Float32, n_pixels)),
        # HW-RT scratch buffers (mirrors hw_primary_ray_buf etc.)
        rays      = MVE.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:n_pixels]),
        hits      = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), n_pixels)),
        n_buf     = MVE.LavaArray(Int32[n_pixels]),
        # Shadow-round scratch (mirrors hw_shadow_*)
        shadow_rays = MVE.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:n_pixels]),
        shadow_hits = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), n_pixels)),
        shadow_n    = MVE.LavaArray(Int32[n_pixels]),
    )
end

const N_PIXELS = 1024
const N_SAMPLES = 4
const MAX_DEPTH = 5
const N_ITERS = 25

println("=== MWE-6: full VolPath shape (8 SoA queues + indirect HW RT) per iter ===")
println("    n_pixels=$N_PIXELS samples=$N_SAMPLES max_depth=$MAX_DEPTH iters=$N_ITERS")

crashed_at = 0
for iter in 1:N_ITERS
    s = alloc_state(N_PIXELS)

    for sample in 1:N_SAMPLES
        v = Float32(iter * sample)

        # Phase 1: generate camera rays (direct dispatch; writes ray_queue + samples)
        gen_camera_rays_kernel!(backend)(s.ray_a.items, s.ray_a.size, s.samples, v;
                                         ndrange=N_PIXELS)

        cur, nxt = s.ray_a, s.ray_b

        # Phase 2: per-depth wavefront loop
        for d in 1:MAX_DEPTH
            # 2a: indirect compute over current ray queue (trace via compute fallback)
            trace_rays_compute_kernel!(backend)(
                cur.items, s.hit_surf.items, s.escaped.items,
                s.hit_surf.size, s.escaped.size, v;
                ndrange=cur.size)

            # 2b: HW indirect RT for primary rays (vkCmdTraceRaysIndirect)
            Mantle.trace_closest_hits_indirect!(s.hits, s.rays, hwtlas.hw_accel, s.n_buf)

            # 2c-2e: medium kernels (indirect dispatches over med_samp / med_scat)
            medium_sample_kernel!(backend)(s.med_samp.items, s.med_scat.items, v;
                                           ndrange=s.med_samp.size)
            medium_direct_lighting_kernel!(backend)(s.med_scat.items, s.shadow.items, v;
                                                    ndrange=s.med_scat.size)
            medium_scatter_kernel!(backend)(s.med_scat.items, nxt.items, s.samples, v;
                                            ndrange=s.med_scat.size)

            # 2f: escaped contribution
            escaped_kernel!(backend)(s.escaped.items, s.pixel_L, v;
                                     ndrange=s.escaped.size)

            # 2g: surface emission + material queue setup
            process_surface_hits_kernel!(backend)(s.hit_surf.items, s.material.items,
                                                  s.pixel_L, v;
                                                  ndrange=s.hit_surf.size)

            # 2h: surface direct lighting
            surface_direct_lighting_kernel!(backend)(s.material.items, s.shadow.items,
                                                     s.samples, v;
                                                     ndrange=s.material.size)

            # 2i: shadow rays (1-3 rounds of HW indirect RT, mirrors hw-rt.jl:359)
            for round in 1:3
                Mantle.trace_closest_hits_indirect!(
                    s.shadow_hits, s.shadow_rays, hwtlas.hw_accel, s.shadow_n)
            end

            # 2j: evaluate materials → next ray queue
            evaluate_materials_kernel!(backend)(s.material.items, nxt.items, s.samples, v;
                                                ndrange=s.material.size)

            # 2k: swap queues
            cur, nxt = nxt, cur
        end

        # Phase 3 & 4: accumulate to rgb + finalize (direct dispatch, n_pixels)
        accumulate_to_rgb_kernel!(backend)(s.pixel_L, s.pixel_rgb, s.weight_sum, v;
                                           ndrange=N_PIXELS)
        finalize_film_kernel!(backend)(s.pixel_rgb, s.weight_sum;
                                       ndrange=N_PIXELS)

        KA.synchronize(backend)
    end

    # Phase 5: drop all per-render state.  Mirror Base.close(vp::VolPath):
    #   vp.state = nothing → GC.gc(false) (synchronous finalizer pass).
    s = nothing
    GC.gc(false)

    if MVE.device_lost(ctx)
        global crashed_at = iter
        break
    end
    print("$iter ")
end
println()
println(crashed_at == 0 ?
    "MWE-6: $N_ITERS iters clean — full VolPath shape doesn't repro." :
    "MWE-6: !!! crashed iter $crashed_at — full shape DOES repro the cascade!")

using Test
@test crashed_at == 0
@test !MVE.device_lost(ctx)
