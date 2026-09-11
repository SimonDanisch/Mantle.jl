# MWE-6L: long-form variant of MWE-6 with per-iter telemetry.
#
# Two questions this answers:
#   1. Does MWE-6's shape eventually crash if we just run more iters?
#      (Earlier "cross-test session limit" pegged the cascade around
#      ~7600 dispatches in one session.)
#   2. What Lava-side state grows monotonically across iters?  If
#      *anything* grows without bound (the retired list, outstanding,
#      unified-arena blocks, timeline gaps), we have a candidate leak.
#
# Telemetry is sampled per iter (cheap) into a CSV-shaped buffer.

using Raycore, Lava, GeometryBasics, StaticArrays, LinearAlgebra, StructArrays, Printf
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx     = Mantle.vk_context()
bq      = ctx.default_bq

hwtlas = Mantle.VulkanTLAS(backend)
mesh = GeometryBasics.normal_mesh(GeometryBasics.Tessellation(
    GeometryBasics.Sphere(GeometryBasics.Point3f(0), 1f0), 8))
push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
Raycore.sync!(hwtlas)

struct RayWI;       a::Float32; b::Float32; c::Float32; flag::Bool;        end
struct MedSampWI;   a::Float32; b::Float32; flag::Bool;                    end
struct MedScatWI;   a::Float32; b::Float32; c::Float32;                    end
struct HitSurfWI;   a::Float32; b::Float32; flag::Bool;                    end
struct MatEvalWI;   a::Float32; b::Float32; c::Float32; d::Float32;        end
struct ShadowWI;    a::Float32; b::Float32; flag::Bool;                    end
struct EscapedWI;   a::Float32; b::Float32; c::Float32;                    end
struct RaySamples;  uc::Float32; u1::Float32; u2::Float32; rr::Float32; u3::Float32; end

struct SoAQueue{T,V,S}
    items::V
    size::S
    capacity::Int32
end
function SoAQueue{T}(n::Int) where T
    fnames = fieldnames(T); ftypes = fieldtypes(T)
    cols = NamedTuple{fnames}(ntuple(i -> Mantle.LavaArray(zeros(ftypes[i], n)), length(fnames)))
    items = StructArray{T}(cols)
    size  = Mantle.LavaArray(Int32[0])
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
    if i <= length(med_samp_q); med_scat_q[i] = MedScatWI(val, val + 1f0, val + 2f0); end
end
@kernel function medium_direct_lighting_kernel!(med_scat_q, shadow_q, val)
    i = @index(Global)
    if i <= length(med_scat_q); shadow_q[i] = ShadowWI(val, val + 1f0, val > 0f0); end
end
@kernel function medium_scatter_kernel!(med_scat_q, ray_q, samples_q, val)
    i = @index(Global)
    if i <= length(med_scat_q)
        s = samples_q[i]; ray_q[i] = RayWI(s.uc + val, s.u1, s.u2, s.rr > 0f0)
    end
end
@kernel function escaped_kernel!(esc_q, pixel_L, val)
    i = @index(Global); if i <= length(esc_q); @inbounds pixel_L[i] += val; end
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
        s = samples_q[i]; shadow_q[i] = ShadowWI(s.uc + val, s.u1, s.rr > 0f0)
    end
end
@kernel function evaluate_materials_kernel!(mat_q, ray_q, samples_q, val)
    i = @index(Global)
    if i <= length(mat_q)
        s = samples_q[i]; ray_q[i] = RayWI(s.uc + val, s.u1, s.u2, s.rr > 0f0)
    end
end
@kernel function accumulate_to_rgb_kernel!(pixel_L, pixel_rgb, weight_sum, val)
    i = @index(Global)
    L = pixel_L[i]
    @inbounds pixel_rgb[i]  += L * val
    @inbounds weight_sum[i] += val
end
@kernel function finalize_film_kernel!(pixel_rgb, weight_sum)
    i = @index(Global); w = weight_sum[i]; if w > 0f0; @inbounds pixel_rgb[i] /= w; end
end

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
        samples  = SoAQueue{RaySamples}(n_pixels).items,
        pixel_L     = Mantle.LavaArray(zeros(Float32, n_pixels)),
        pixel_rgb   = Mantle.LavaArray(zeros(Float32, n_pixels)),
        weight_sum  = Mantle.LavaArray(zeros(Float32, n_pixels)),
        rays      = Mantle.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:n_pixels]),
        hits      = Mantle.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), n_pixels)),
        n_buf     = Mantle.LavaArray(Int32[n_pixels]),
        shadow_rays = Mantle.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3) for _ in 1:n_pixels]),
        shadow_hits = Mantle.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), n_pixels)),
        shadow_n    = Mantle.LavaArray(Int32[n_pixels]),
    )
end

function render_one_iter!(s, hwtlas, iter::Int; n_samples=4, max_depth=5, n_pixels=1024)
    for sample in 1:n_samples
        v = Float32(iter * sample)
        gen_camera_rays_kernel!(backend)(s.ray_a.items, s.ray_a.size, s.samples, v;
                                         ndrange=n_pixels)
        cur, nxt = s.ray_a, s.ray_b
        for _ in 1:max_depth
            trace_rays_compute_kernel!(backend)(
                cur.items, s.hit_surf.items, s.escaped.items,
                s.hit_surf.size, s.escaped.size, v; ndrange=cur.size)
            Mantle.trace_closest_hits_indirect!(s.hits, s.rays, hwtlas.hw_accel, s.n_buf)
            medium_sample_kernel!(backend)(s.med_samp.items, s.med_scat.items, v;
                                           ndrange=s.med_samp.size)
            medium_direct_lighting_kernel!(backend)(s.med_scat.items, s.shadow.items, v;
                                                    ndrange=s.med_scat.size)
            medium_scatter_kernel!(backend)(s.med_scat.items, nxt.items, s.samples, v;
                                            ndrange=s.med_scat.size)
            escaped_kernel!(backend)(s.escaped.items, s.pixel_L, v;
                                     ndrange=s.escaped.size)
            process_surface_hits_kernel!(backend)(s.hit_surf.items, s.material.items,
                                                  s.pixel_L, v; ndrange=s.hit_surf.size)
            surface_direct_lighting_kernel!(backend)(s.material.items, s.shadow.items,
                                                     s.samples, v; ndrange=s.material.size)
            for _ in 1:3
                Mantle.trace_closest_hits_indirect!(
                    s.shadow_hits, s.shadow_rays, hwtlas.hw_accel, s.shadow_n)
            end
            evaluate_materials_kernel!(backend)(s.material.items, nxt.items, s.samples, v;
                                                ndrange=s.material.size)
            cur, nxt = nxt, cur
        end
        accumulate_to_rgb_kernel!(backend)(s.pixel_L, s.pixel_rgb, s.weight_sum, v;
                                           ndrange=n_pixels)
        finalize_film_kernel!(backend)(s.pixel_rgb, s.weight_sum; ndrange=n_pixels)
        KA.synchronize(backend)
    end
end

function snapshot(bq, ctx, iter)
    return (
        iter = iter,
        device_lost   = Mantle.device_lost(ctx),
        next_tl       = Int(Mantle.driver(bq).next_timeline),
        outstanding   = length(bq.outstanding),
        free = length(bq.free),
        deferred      = length(bq.pending),
        deferred_as   = length(bq.retiring),
        # What a recording reads: blocks of the unified arena, and how many
        # regions of them are still handed out. Growth in either is the leak
        # this MWE is looking for; the two slab rings it used to read are gone.
        blocks        = length(Mantle.unifiedblocks(ctx)),
        live_regions  = sum(b -> length(b.live), Mantle.unifiedblocks(ctx); init = 0),
    )
end

const N_ITERS = 200
println("=== MWE-6L: long-form ($N_ITERS iters) with per-iter telemetry ===")

snapshots = NamedTuple[]
push!(snapshots, snapshot(bq, ctx, 0))

crashed_at = 0
for iter in 1:N_ITERS
    s = alloc_state(1024)
    render_one_iter!(s, hwtlas, iter)
    s = nothing
    GC.gc(false)

    if iter % 10 == 0 || iter <= 3
        push!(snapshots, snapshot(bq, ctx, iter))
    end

    if Mantle.device_lost(ctx)
        global crashed_at = iter
        push!(snapshots, snapshot(bq, ctx, iter))
        break
    end
    if iter % 25 == 0; print("$iter "); end
end
println()

println("\n--- per-iter snapshots ---")
println("iter  device_lost  next_tl  outstanding  free  deferred  def_as  unified blocks/live")
for s in snapshots
    @printf "%4d  %-11s  %7d  %11d  %12d  %8d  %6d  %19s\n" s.iter string(s.device_lost) s.next_tl s.outstanding s.free s.deferred s.deferred_as "$(s.blocks)/$(s.live_regions)"
end

if crashed_at == 0
    println("\nMWE-6L: $N_ITERS iters clean.  No monotonic growth seen → cascade is not in this shape.")
else
    println("\nMWE-6L: !!! crashed at iter $crashed_at — telemetry above shows the trajectory.")
end

nothing
