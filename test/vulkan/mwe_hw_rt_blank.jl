# mwe_hw_rt_blank.jl — minimal reproducer for the hw_accel=true blank-render
# bug observed in RayDemo Crown/bunny_cloud on Windows AMDVLK.
#
# Scene: single triangle at z=5.
# Kernel: 64 threads, each fires a ray from a grid origin in +z direction,
#         calls `Raycore.closest_hit(accel, ray)` via inline ray query, writes
#         hit_t (or -1 on miss) to an output buffer.
#
# CPU reference computes the expected hit_t with Möller–Trumbore.
#
# We run TWICE:
#   1. Default multi-OpFunction emission (current Lava sd/no-inline state)
#   2. With `force_inline_all=true` via FORCE_INLINE_KERNEL_PATTERNS hook
# and compare results to the CPU reference.
#
# Expected outcome on Windows AMDVLK before the fix:
#   Default → all -1 (blank)
#   Inlined → matches CPU reference
#
# Run:  julia --project=. dev/Lava/test/mwe_hw_rt_blank.jl

using Lava, Raycore, Adapt
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I, cross, dot
using StaticArrays: SMatrix
const Mat4f = SMatrix{4, 4, Float32, 16}

struct RayQueryAccel{T, O, Tri} <: Raycore.AbstractAdaptedAccel
    triangles::T
    offsets::O
    empty::Tri
end
Adapt.adapt_structure(to, a::RayQueryAccel) = RayQueryAccel(
    Adapt.adapt(to, a.triangles),
    Adapt.adapt(to, a.offsets),
    a.empty,
)

@inline function _rq_collect(accel::RayQueryAccel)
    while Lava.lava_ray_query_proceed()
    end
    kind = Lava.lava_ray_query_get_type(true)
    kind != UInt32(1) && return (false, accel.empty, 0f0, (0f0, 0f0, 0f0), UInt32(0))
    t = Lava.lava_ray_query_get_t(true)
    inst_id = Lava.lava_ray_query_get_instance_id(true)
    inst_custom = Lava.lava_ray_query_get_instance_custom_index(true)
    prim_idx = Lava.lava_ray_query_get_primitive_index(true)
    bx, by = Lava.lava_ray_query_get_barycentrics(true)
    @inbounds tri_idx = Int(accel.offsets[inst_id + UInt32(1)]) + Int(prim_idx) + 1
    @inbounds tri = accel.triangles[tri_idx]
    return (true, tri, t, (1f0 - bx - by, bx, by), inst_custom)
end

@inline function Raycore.closest_hit(accel::RayQueryAccel, ray::Raycore.AbstractRay)
    o = ray.o; d = ray.d
    Lava.lava_ray_query_init(UInt32(0), UInt32(0xFF),
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return _rq_collect(accel)
end

function ray_tri_t(o, d, v0, v1, v2)
    e1 = v1 - v0; e2 = v2 - v0
    h = cross(d, e2); a = dot(e1, h)
    abs(a) < 1f-7 && return -1f0
    f = 1f0 / a
    s = o - v0
    u = f * dot(s, h)
    (u < 0f0 || u > 1f0) && return -1f0
    q = cross(s, e1)
    v = f * dot(d, q)
    (v < 0f0 || u + v > 1f0) && return -1f0
    t = f * dot(e2, q)
    return t < 0f0 ? -1f0 : t
end

function run_once(label::String)
    println("\n=== $label ===")
    backend = Mantle.LavaBackend()
    bq = backend.bq

    tri_v0 = Point3f(-1f0, -1f0, 5f0)
    tri_v1 = Point3f( 1f0, -1f0, 5f0)
    tri_v2 = Point3f( 0f0,  1f0, 5f0)
    verts = [tri_v0, tri_v1, tri_v2]
    faces = [GLTriangleFace(1, 2, 3)]
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, mesh, Mat4f(I))
    Raycore.sync!(hwtlas)
    Tri = eltype(eltype(hwtlas.blas_triangles))
    accel = RayQueryAccel(hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri))

    n = 64
    origins = [Point3f((i%8 - 3.5f0) * 0.25f0, (i÷8 - 3.5f0) * 0.25f0, 0f0) for i in 0:n-1]
    ref = Float32[
        let t = ray_tri_t(origins[i], Vec3f(0,0,1), tri_v0, tri_v1, tri_v2)
            t < 0f0 ? -1f0 : t
        end
        for i in 1:n
    ]

    function k(t_out::Lava.LavaDeviceArray{Float32, 1},
                origins::Lava.LavaDeviceArray{Point3f, 1},
                accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        hit, _prim, t, _bary, _icx = Raycore.closest_hit(accel, ray)
        @inbounds t_out[i] = hit ? t : -1f0
        return nothing
    end

    t_out = Mantle.LavaArray(fill(-2f0, n))
    origins_g = Mantle.LavaArray(origins)
    Mantle.lava_launch!(bq, k, t_out, origins_g, accel;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    gpu_t = Array(t_out)

    n_hit_ref = count(>=(0f0), ref)
    n_hit_gpu = count(>=(0f0), gpu_t)
    n_match = count(1:n) do i
        ref[i] < 0f0 ? gpu_t[i] < 0f0 : isapprox(gpu_t[i], ref[i]; atol=1f-3)
    end
    println("  ref:  $n_hit_ref hits / $n rays  (expected t ≈ 5.0 inside triangle, -1 outside)")
    println("  gpu:  $n_hit_gpu hits / $n rays")
    println("  match: $n_match / $n")
    println("  sample gpu[1..8]: ", round.(gpu_t[1:8]; digits=2))
    println("  sample ref[1..8]: ", round.(ref[1:8]; digits=2))
    return n_match == n
end

function try_run(label)
    try
        return run_once(label)
    catch e
        println("  ✗ CRASHED: ", first(sprint(showerror, e), 300))
        return false
    end
end

# ── Run 1: default multi-OpFunction emission ─────────────────────────────
empty!(Lava.FORCE_INLINE_KERNEL_PATTERNS)
Mantle.clear_spirv_disk_cache!()
empty!(Mantle.vk_context().caches.linked)
default_ok = try_run("default (multi-OpFunction)")

# Reset between runs in case of crash
try Mantle.reset_device!() catch e; @warn "reset failed: $(first(sprint(showerror,e),100))" end

# ── Run 2: force_inline_all=true via the debug hook ──────────────────────
empty!(Lava.FORCE_INLINE_KERNEL_PATTERNS)
push!(Lava.FORCE_INLINE_KERNEL_PATTERNS, "")  # empty matches everything
Mantle.clear_spirv_disk_cache!()
empty!(Mantle.vk_context().caches.linked)
inlined_ok = try_run("force_inline_all=true")

println("\n────────────────────────────────────────")
println("default (multi-OpFunction):  ", default_ok ? "PASS" : "FAIL")
println("force_inline_all=true:       ", inlined_ok ? "PASS" : "FAIL")
println("────────────────────────────────────────")
if !default_ok && inlined_ok
    println("✗ Reproduced: hw_accel multi-OpFunction is broken; inlining fixes it.")
elseif default_ok && inlined_ok
    println("Both pass — bug not reproducible here. Investigate further.")
elseif !inlined_ok
    println("Both fail — bug is NOT in multi-OpFunction; something else is wrong.")
end
