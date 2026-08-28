# ==============================================================================
# HW TLAS stress + correctness test (Lava backend)
# ==============================================================================
#
# Lava is a hard test dep for Raycore, so this runs as part of the normal suite
# (a real Vulkan device must be available). Exercises two failure modes the
# HWTLAS + RaycoreLavaExt path is especially prone to:
#
# 1. **Correctness** — every `push!` / `delete!` / `update_transform!` /
#    `update_transforms!` followed by `sync!(hwtlas)` must reflect in
#    subsequent `trace_closest_hits!` results. If the Vulkan TLAS / BLAS
#    / per-instance offset buffers retain stale BDA captures from before
#    the mutation, rays will hit the OLD geometry → wrong `t` / wrong
#    primitive id. We compare each batch of hits against an analytic CPU
#    reference that knows where the triangles *should* be after the
#    transform.
#
# 2. **Leak / UAF under stress** — the rebuild cycle allocates fresh
#    `LavaTLAS`, `LavaBLAS`, `tri_gpu`, `off_gpu` objects on every dirty
#    `sync!`. The extension's `release_hw_accel_state!` + Lava's
#    finalizer + pool-reuse path must drop the old ones; otherwise
#    `GPU_LIVE_BYTES`, the pool-block count, and the `LIVE_BUFFERS` set
#    grow without bound. We hammer the TLAS with ≥500 rebuild cycles and
#    assert every counter stays under a tight ceiling relative to
#    baseline. If any GPU buffer that was freed mid-cycle was still
#    referenced by a pending dispatch, a hit result would come back with
#    nonsense values (or the device would be lost) — the correctness
#    pass under the same iterations catches that.
# ==============================================================================

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Lava, Mantle
using Raycore
using KernelAbstractions
using Random
const KA = KernelAbstractions

# ------------------------------------------------------------------------------
# Scene helpers
# ------------------------------------------------------------------------------

"""Single-triangle mesh at z=0: verts (0,0,0), (1,0,0), (0,1,0)."""
function unit_triangle_mesh()
    verts = [Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0)]
    faces = [GLTriangleFace(1,2,3)]
    return GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
end

"""Axis-aligned box mesh (GeometryBasics)."""
box_mesh(origin::Vec3f, extent::Vec3f) =
    GeometryBasics.normal_mesh(Rect3f(origin, extent))

# Shared HW-TLAS helpers — see test/hwtlas_helpers.jl (provides `translation`).
isdefined(@__MODULE__, :translation) ||
    include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

"""Translation as `Mat3x4f` (Vulkan row-major 3×4) for `update_transform!`."""
vk_translation(dx, dy, dz) = Mantle.mat4_to_vk_transform(translation(dx, dy, dz))

"""Downward ray hits a z=0 triangle translated by `offset` iff the ray's
xy point lies inside the translated triangle; returns the analytic t."""
function analytic_hit(ray_ox, ray_oy, ray_oz, offset::NTuple{3,Float32})
    ox, oy = ray_ox - offset[1], ray_oy - offset[2]
    # Point-in-triangle test for the unit-triangle (0,0), (1,0), (0,1):
    inside = (ox >= 0f0) && (oy >= 0f0) && (ox + oy <= 1f0)
    inside ? (true, ray_oz - offset[3]) : (false, 0f0)
end

# ------------------------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------------------------

"""Build an HWTLAS on Lava with `N` instances of the unit triangle, each
placed at a distinct translation so a single vertical ray per instance
hits each one exactly once."""
function build_scene(N::Int)
    hwtlas = Mantle.HWTLAS(LavaBackend())
    mesh = unit_triangle_mesh()
    handles = Raycore.TLASHandle[]
    offsets = NTuple{3,Float32}[]
    for i in 1:N
        off = (Float32(2i), 0f0, 0f0)  # spaced 2 apart in x, z=0
        T = translation(off...)
        h = push!(hwtlas, mesh, T; instance_id=UInt32(i))
        push!(handles, h)
        push!(offsets, off)
    end
    Raycore.sync!(hwtlas)
    return hwtlas, handles, offsets
end

"""Build test rays: one ray per instance, dropping from z=+1 straight
down through the triangle centroid at x=offset.x+0.25, y=0.25."""
function make_rays(offsets::Vector{NTuple{3,Float32}})
    rays = [Raycore.RTRay(ox + 0.25f0, 0.25f0, 1f0, 0f0, 0f0, 0f0, -1f0, 1f3)
            for (ox, _, _) in offsets]
    return rays
end

"""Upload rays + run HW trace and return the hit result vector (on CPU).
Uses the `HardwareAccel` already built by `Raycore.sync!(hwtlas)` — calling
`Mantle.HardwareAccel(hwtlas)` would rebuild the same thing from scratch via
the generic CPU-TLAS path and fails because `hwtlas.instances` is a
lightweight length-only shim, not a real instance vector."""
function trace_rays_cpu(hwtlas, offsets)
    rays   = make_rays(offsets)
    n      = length(rays)
    gpu_r  = LavaArray(rays)
    gpu_h  = LavaArray(fill(Raycore.RTHitResult(0, 0, 0, 0, 0, 0, 0, 0), n))
    Mantle.trace_closest_hits!(gpu_h, gpu_r, hwtlas.hw_accel, n)
    return Array(gpu_h)
end

# ------------------------------------------------------------------------------
# Correctness test
# ------------------------------------------------------------------------------

"""Assert that every ray hit matches the analytic expectation for the
current set of `offsets` (one per instance). Used after each mutation."""
function assert_hits_match(hits::Vector{Raycore.RTHitResult},
                           offsets::Vector{NTuple{3,Float32}})
    @assert length(hits) == length(offsets)
    all_good = true
    for i in eachindex(offsets)
        off = offsets[i]
        (want_hit, want_t) = analytic_hit(Float32(off[1] + 0.25f0),
                                           Float32(0.25f0),
                                           Float32(1),
                                           off)
        h = hits[i]
        got_hit = h.hit != UInt32(0)
        if got_hit != want_hit
            @warn "instance $i: hit mismatch" got_hit want_hit offset=off
            all_good = false
            continue
        end
        if want_hit && !isapprox(Float32(h.t), want_t; atol=1f-3)
            @warn "instance $i: t mismatch" got_t=h.t want_t offset=off
            all_good = false
        end
    end
    return all_good
end

@testset "HW TLAS — correctness under mutation" begin
    N = 8
    hwtlas, handles, offsets = build_scene(N)

    # Baseline: every instance hits as expected.
    @test assert_hits_match(trace_rays_cpu(hwtlas, offsets), offsets)

    # Mutate every transform with random translations; confirm hits follow.
    for iter in 1:20
        for i in 1:N
            new_off = (Float32(2i + 0.1 * iter),
                       Float32(0.3 * sinpi(iter / 5)),
                       Float32(-0.05 * iter))
            offsets[i] = new_off
            @assert Raycore.update_transform!(hwtlas, handles[i],
                                              vk_translation(new_off...))
        end
        Raycore.sync!(hwtlas)
        @test assert_hits_match(trace_rays_cpu(hwtlas, offsets), offsets)
    end

    # Delete half, re-push with new transforms; topology change path.
    kept_handles  = handles[1:2:end]
    kept_offsets  = offsets[1:2:end]
    for h in handles[2:2:end]
        Raycore.delete!(hwtlas, h)
    end
    # Re-push fresh instances: same mesh, new offsets.
    new_offsets = [(Float32(100 + 2i), Float32(1), Float32(0.2 * i))
                   for i in 1:(N÷2)]
    for off in new_offsets
        h = push!(hwtlas, unit_triangle_mesh(), translation(off...);
                  instance_id=UInt32(99))
        push!(kept_handles, h)
        push!(kept_offsets, off)
    end
    Raycore.sync!(hwtlas)
    @test assert_hits_match(trace_rays_cpu(hwtlas, kept_offsets), kept_offsets)
end

# ------------------------------------------------------------------------------
# Stress / leak test
# ------------------------------------------------------------------------------

function snapshot_state()
    gpu_bytes  = Mantle.gpu_live_bytes()
    n_buffers  = Mantle.live_buffer_count()
    n_pool     = length(Mantle.poolblocks(Mantle.vk_context()))
    (gpu_bytes=gpu_bytes, live_bufs=n_buffers, pool_blocks=n_pool)
end

@testset "HW TLAS — stress / leak bounds" begin
    N = 16
    hwtlas, handles, offsets = build_scene(N)

    # Warm up: do a couple of rebuild cycles + a trace, then GC. The
    # baseline we lock onto is the *post-warmup* state; the first few
    # iterations legitimately populate pools, kernel caches, SBT, etc.
    for _ in 1:4
        for i in 1:N
            offsets[i] = (Float32(2i), Float32(0.1), 0f0)
            Raycore.update_transform!(hwtlas, handles[i], vk_translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        _ = trace_rays_cpu(hwtlas, offsets)
    end
    GC.gc(true); GC.gc(true)
    baseline = snapshot_state()
    @info "baseline" baseline

    # Hammer the rebuild cycle. Every iteration:
    #   * shuffle per-instance transforms (push fresh NTuple{12,Float32})
    #   * call sync! — rebuilds TLAS (and maybe BLAS offsets) on Lava
    #   * trace + verify — catches use-after-free and UAF-masked noise
    n_iters   = 500
    max_hits  = 0  # debugging aid — peak samples during the loop
    for iter in 1:n_iters
        for i in 1:N
            offsets[i] = (Float32(2i + 0.01 * iter),
                          Float32(0.2 * cospi(iter / 7)),
                          Float32(-0.05 * sinpi(iter / 9)))
            Raycore.update_transform!(hwtlas, handles[i], vk_translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        hits = trace_rays_cpu(hwtlas, offsets)
        @assert assert_hits_match(hits, offsets) "iter $iter: hits diverged — UAF suspected"
        max_hits = max(max_hits, count(h -> h.hit != 0, hits))
    end
    GC.gc(true); GC.gc(true)
    final = snapshot_state()
    @info "after $n_iters iterations" final peak_hits=max_hits

    # Allow small growth (pool block fragmentation, kernel caches warming up),
    # but the delta must not scale with iteration count. These ceilings are
    # intentionally tight — anything looser stops being a real leak test.
    @testset "no unbounded GPU memory growth" begin
        @test final.gpu_bytes   <= baseline.gpu_bytes + 256 * 1024^2  # +256 MiB
        @test final.live_bufs   <= baseline.live_bufs + 16
        @test final.pool_blocks <= baseline.pool_blocks + 8
    end
end

# ------------------------------------------------------------------------------
# update_transforms! — bulk transform update via GPU kernel
# ------------------------------------------------------------------------------

@testset "HW TLAS — update_transforms! (CPU input) refits without rebuild" begin
    N = 8
    hwtlas, handles, offsets = build_scene(N)

    # Replace the per-handle batch with a single multi-instance handle so
    # we can exercise update_transforms! with a length>1 transforms vector.
    for h in handles
        Raycore.delete!(hwtlas, h)
    end
    init_xfs = [translation(Float32(2i), 0f0, 0f0) for i in 1:N]
    multi_handle = push!(hwtlas, unit_triangle_mesh(), init_xfs;
                         instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)
    pinned_hw_tlas = hwtlas.hw_tlas

    # Drive update_transforms! 20 times; sync! must take the refit path
    # (same hw_tlas identity) every time.
    for iter in 1:20
        new_xfs = Mantle.LavaArray([vk_translation(Float32(2i + 0.05 * iter), 0f0, 0f0) for i in 1:N])
        Raycore.update_transforms!(hwtlas, multi_handle, new_xfs)
        @test hwtlas.transforms_dirty == true
        @test hwtlas.dirty == false
        Raycore.sync!(hwtlas)
        @test hwtlas.transforms_dirty == false
        @test hwtlas.hw_tlas === pinned_hw_tlas
    end

    # Final correctness check — rays at the post-update positions must hit.
    final_offsets = [(Float32(2i + 0.05 * 20), 0f0, 0f0) for i in 1:N]
    @test assert_hits_match(trace_rays_cpu(hwtlas, final_offsets), final_offsets)
end

@testset "HW TLAS — update_transforms! accepts LavaArray input" begin
    N = 4
    init_xfs = [translation(Float32(2i), 0f0, 0f0) for i in 1:N]
    hwtlas = Mantle.HWTLAS(LavaBackend())
    h = push!(hwtlas, unit_triangle_mesh(), init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    new_xfs_cpu = [vk_translation(Float32(2i + 5f0), 0f0, 0f0) for i in 1:N]
    new_xfs_gpu = Mantle.LavaArray(new_xfs_cpu)
    Raycore.update_transforms!(hwtlas, h, new_xfs_gpu)
    @test hwtlas.transforms_dirty == true
    Raycore.sync!(hwtlas)
    @test hwtlas.transforms_dirty == false

    final_offsets = [(Float32(2i + 5f0), 0f0, 0f0) for i in 1:N]
    @test assert_hits_match(trace_rays_cpu(hwtlas, final_offsets), final_offsets)
end

@testset "HW TLAS — update_transforms! then delete!(handle) is safe" begin
    N = 4
    init_xfs = [translation(Float32(2i), 0f0, 0f0) for i in 1:N]
    hwtlas = Mantle.HWTLAS(LavaBackend())
    h_a = push!(hwtlas, unit_triangle_mesh(), init_xfs; instance_mask=UInt8(0xff))
    h_b = push!(hwtlas, unit_triangle_mesh(), translation(20f0, 0f0, 0f0))
    Raycore.sync!(hwtlas)

    new_xfs = Mantle.LavaArray([vk_translation(Float32(2i + 1f0), 0f0, 0f0) for i in 1:N])
    Raycore.update_transforms!(hwtlas, h_a, new_xfs)
    # Delete BEFORE syncing the refit -- topology change wins.
    @test Raycore.delete!(hwtlas, h_a) == true
    @test hwtlas.dirty == true
    Raycore.sync!(hwtlas)
    @test Raycore.n_instances(hwtlas) == 1

    # Surviving batch still traces.
    @test assert_hits_match(trace_rays_cpu(hwtlas, [(20f0, 0f0, 0f0)]), [(20f0, 0f0, 0f0)])
end

# ------------------------------------------------------------------------------
# Interleaved bulk update + trace at scale (refit path) — analogous to the
# Raycore SW stress, but the refit goes through MODE_UPDATE_KHR on the
# Vulkan AS and the writes happen via a GPU kernel into `instance_buf`.
# Catches: per-frame staleness on the HW path, GPU-kernel-vs-AS-build
# timeline ordering bugs, refit-vs-rebuild misclassification at high count.
# ------------------------------------------------------------------------------

@testset "HW TLAS — interleaved update_transforms! + trace tight loop (1000 inst, refit)" begin
    N = 1000
    hwtlas = Mantle.HWTLAS(LavaBackend())
    mesh = unit_triangle_mesh()
    init_xfs = [translation(Float32(2i), 0f0, 0f0) for i in 1:N]
    h = push!(hwtlas, mesh, init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    # Pin LavaTLAS identity — refit must keep the same hw_tlas across all
    # frames.  If sync! ever drops to rebuild for a frame (because the dirty
    # flag was wrong) this assertion trips.
    pinned_hw_tlas = hwtlas.hw_tlas

    n_frames = 50
    for frame in 1:n_frames
        # Move every instance: x_i = 2i + 0.05*frame, y = small oscillation.
        new_xfs = Mantle.LavaArray([vk_translation(Float32(2i + 0.05 * frame),
                                               Float32(0.1 * sinpi(frame / 7)),
                                               0f0)
                                  for i in 1:N])
        new_offsets = [(Float32(2i + 0.05 * frame),
                        Float32(0.1 * sinpi(frame / 7)),
                        0f0)
                       for i in 1:N]

        Raycore.update_transforms!(hwtlas, h, new_xfs)
        @test hwtlas.transforms_dirty == true
        @test hwtlas.dirty == false
        Raycore.sync!(hwtlas)
        @test hwtlas.transforms_dirty == false
        @test hwtlas.hw_tlas === pinned_hw_tlas    # refit, not rebuild

        # Trace every instance THIS frame; hit positions must reflect THIS
        # frame's transforms (not the previous frame's).
        hits = trace_rays_cpu(hwtlas, new_offsets)
        @test assert_hits_match(hits, new_offsets)
    end
end

# ------------------------------------------------------------------------------
# Interleaved rebuild + trace at scale (topology change every frame).
# Forces full LavaTLAS rebuild per frame; old hw_tlas / instance buffers
# must be timeline-deferred-freed AFTER this frame's trace completes —
# otherwise the trace reads from a freed buffer and we see wrong t-values
# or device fault.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 500-iter mesh grow/shrink with HW raytracing every iter (correctness + leak)
# ------------------------------------------------------------------------------
#
# HW analogue of Raycore's grow/shrink test.  Each iter: drop the BLAS,
# build a fresh one with a new tessellation, sync (full LavaTLAS rebuild,
# old hw_tlas timeline-deferred-freed), then HW raytrace through the new
# geometry. Catches:
#   * stale BLAS device-address captures the AS retains across rebuilds
#   * old instance_buf / tri_gpu / off_gpu being freed before the previous
#     iter's trace dispatch completed (UAF on the previous frame's reads)
#   * pool-block / live_buf accumulation tied to per-iter mesh size churn

"""Analytic hit for a downward ray at (0, 0, 5) against a unit sphere at
(0, 0, offset_z): top of sphere at z=offset_z+1, t = 5 - offset_z - 1."""
sphere_hit_t(offset_z) = 5f0 - Float32(offset_z) - 1f0

# Sphere BLAS test scaffolding -- different geometry from unit_triangle_mesh,
# kept separate so the existing triangle-based tests aren't disturbed.
sphere_mesh_n(n::Int) = GeometryBasics.normal_mesh(
    GeometryBasics.Tessellation(GeometryBasics.Sphere(GeometryBasics.Point3f(0,0,0), 1f0), n))

function trace_one_sphere(hwtlas, ox, oy, oz)
    # Single ray (ox, oy, oz) → -z direction.  Returns (hit, t).
    rays  = [Raycore.RTRay(Float32(ox), Float32(oy), Float32(oz),
                           0f0, 0f0, 0f0, -1f0, 1f3)]
    gpu_r = LavaArray(rays)
    gpu_h = LavaArray([Raycore.RTHitResult(0, 0, 0, 0, 0, 0, 0, 0)])
    Mantle.trace_closest_hits!(gpu_h, gpu_r, hwtlas.hw_accel, 1)
    h = Array(gpu_h)[1]
    return (hit = h.hit != UInt32(0), t = Float32(h.t))
end

# Shared schedule (mirrors Raycore's): 5 cycles of 100 iters, peaks 16..96.
function hw_grow_shrink_tess(iter::Int)
    cycle_len = 100
    peaks     = (16, 32, 48, 64, 96)
    cycle_i   = ((iter - 1) ÷ cycle_len) % length(peaks) + 1
    peak      = peaks[cycle_i]
    phase     = (iter - 1) % cycle_len
    half      = cycle_len ÷ 2
    if phase < half
        max(8, Int(round(8 + (peak - 8) * (phase / half))))
    else
        max(8, Int(round(peak - (peak - 8) * ((phase - half) / half))))
    end
end

@testset "HW TLAS — 500-iter mesh grow/shrink + HW trace per iter" begin
    hwtlas = Mantle.HWTLAS(LavaBackend())
    h = push!(hwtlas, sphere_mesh_n(8), translation(0, 0, 0); instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    n_iters = 500
    saw_min, saw_max = typemax(Int), 0
    for iter in 1:n_iters
        tess  = hw_grow_shrink_tess(iter)
        saw_min, saw_max = min(saw_min, tess), max(saw_max, tess)
        z_off = Float32((iter % 30) * 0.05)            # 0 .. 1.45 — ray always reaches

        Raycore.delete!(hwtlas, h)
        h = push!(hwtlas, sphere_mesh_n(tess), translation(0, 0, z_off);
                  instance_mask=UInt8(0xff))
        Raycore.sync!(hwtlas)

        @test Raycore.n_instances(hwtlas) == 1
        @test length(hwtlas.instance_batches) == 1

        # HW trace through the freshly-built BVH.  If the previous iter's
        # trace dispatch hadn't completed before sync! freed the old TLAS
        # backing (instance_buf, hw_tlas, tri_gpu, off_gpu), this trace
        # would either fault or return wrong values — we'd see the t
        # mismatch immediately and not 500 iters from now.
        r = trace_one_sphere(hwtlas, 0f0, 0f0, 5f0)
        @test r.hit
        @test isapprox(r.t, sphere_hit_t(z_off); atol=0.15f0)
    end

    @test saw_min <= 10
    @test saw_max >= 90
end

@testset "HW TLAS — interleaved delete+push+sync+trace tight loop (500 inst, rebuild)" begin
    N = 500
    hwtlas = Mantle.HWTLAS(LavaBackend())
    mesh = unit_triangle_mesh()
    init_xfs = [translation(Float32(2i), 0f0, 0f0) for i in 1:N]
    h = push!(hwtlas, mesh, init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    n_frames = 30
    for frame in 1:n_frames
        # Drop the whole batch and re-push — full topology rebuild per frame.
        Raycore.delete!(hwtlas, h)
        new_xfs = [translation(Float32(2i + 0.05 * frame),
                                Float32(0.1 * cospi(frame / 5)),
                                0f0)
                   for i in 1:N]
        new_offsets = [(Float32(2i + 0.05 * frame),
                        Float32(0.1 * cospi(frame / 5)),
                        0f0)
                       for i in 1:N]
        h = push!(hwtlas, mesh, new_xfs; instance_mask=UInt8(0xff))
        Raycore.sync!(hwtlas)

        @test Raycore.n_instances(hwtlas) == N
        @test length(hwtlas.instance_batches) == 1

        # Trace + verify against THIS frame's geometry.  A UAF on the freed
        # previous-frame instance_buf or BLAS would manifest as wrong hits
        # here (or a device fault, which the testset can't recover from).
        hits = trace_rays_cpu(hwtlas, new_offsets)
        @test assert_hits_match(hits, new_offsets)
    end
end

@testset "HW TLAS — n_instances matches live batch count under churn" begin
    rng = Random.MersenneTwister(0xCAFEBABE)
    hwtlas = Mantle.HWTLAS(LavaBackend())
    handles = Raycore.TLASHandle[]
    expected = 0
    for iter in 1:50
        op = rand(rng, 1:3)
        if op == 1 && length(handles) < 12
            n = rand(rng, 1:5)
            xfs = [translation(Float32(2i + iter), 0f0, 0f0) for i in 1:n]
            h = push!(hwtlas, unit_triangle_mesh(), xfs)
            push!(handles, h)
            expected += n
        elseif op == 2 && length(handles) > 0
            i = rand(rng, 1:length(handles))
            h = handles[i]
            n_h = hwtlas.instance_batches[hwtlas.handle_to_batch_idx[h]].n
            Raycore.delete!(hwtlas, h)
            deleteat!(handles, i)
            expected -= n_h
        else
            # No-op churn.
        end
        if iter % 5 == 0
            Raycore.sync!(hwtlas)
        end
        @test Raycore.n_instances(hwtlas) == expected
    end
end

println("\nAll HW TLAS tests passed.")
