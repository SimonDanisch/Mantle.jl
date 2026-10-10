# ==============================================================================
# HW HWTLAS stress + correctness test
# ==============================================================================
#
# Runs on every backend that builds a hardware HWTLAS. Exercises two failure
# modes the hardware path is especially prone to:
#
# 1. **Correctness** — every `push!` / `delete!` / `update_transform!` /
#    `update_transforms!` followed by `sync!(hwtlas)` must reflect in
#    subsequent traces. If the TLAS, a BLAS or the per-instance offset buffers
#    retain stale device addresses from before the mutation, rays will hit the
#    OLD geometry → wrong `t` / wrong primitive. We compare each batch of hits
#    against a host reference that knows where the triangles *should* be after
#    the transform.
#
# 2. **Leak / UAF under stress** — the rebuild cycle allocates a fresh top-level
#    structure, fresh BLASes and fresh triangle/offset arrays on every dirty
#    `sync!`. The old ones must be released once nothing in flight reads them;
#    otherwise the device pool's bytes, blocks and regions on loan grow without
#    bound. We hammer the HWTLAS with ≥500 rebuild cycles and assert every
#    counter stays under a tight ceiling relative to baseline. If any GPU buffer
#    that was freed mid-cycle was still referenced by a pending trace, a hit
#    would come back with nonsense values (or the device would be lost) — the
#    correctness pass under the same iterations catches that.
#
# Traces go through `test/hwtlas_helpers.jl`: `Raycore.closest_hit` in a kernel.
# ==============================================================================

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Mantle
using Raycore
using Adapt
using KernelAbstractions
using Random
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const Tri = Raycore.Triangle{UInt32}

# ------------------------------------------------------------------------------
# Scene helpers
# ------------------------------------------------------------------------------

"""The unit triangle's vertices: (0,0,0), (1,0,0), (0,1,0)."""
const UNIT_TRI = (Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0))

"""Single-triangle mesh at z=0: verts (0,0,0), (1,0,0), (0,1,0)."""
function unit_triangle_mesh()
    faces = [GLTriangleFace(1,2,3)]
    return GeometryBasics.normal_mesh(GeometryBasics.Mesh(collect(UNIT_TRI), faces))
end

"""Translation as `Mat3x4f` (row-major 3×4), the form an instance stores, for
`update_transform!`."""
mat3x4translation(dx, dy, dz) = Mantle.mat4_to_vk_transform(tlastranslation(dx, dy, dz))

# ------------------------------------------------------------------------------
# Shared setup
# ------------------------------------------------------------------------------

"""Build an HWTLAS with `N` instances of the unit triangle, each placed at a
distinct translation so a single vertical ray per instance hits each one
exactly once."""
function build_scene(N::Int)
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    mesh = unit_triangle_mesh()
    handles = Raycore.TLASHandle[]
    offsets = NTuple{3,Float32}[]
    for i in 1:N
        off = (Float32(2i), 0f0, 0f0)  # spaced 2 apart in x, z=0
        T = tlastranslation(off...)
        h = push!(hwtlas, mesh, T; instance_id=UInt32(i))
        push!(handles, h)
        push!(offsets, off)
    end
    Raycore.sync!(hwtlas)
    return hwtlas, handles, offsets
end

"""One ray per instance, dropping from z=+1 straight down through
x=offset.x+0.25, y=0.25."""
ray_origins(offsets) = [Point3f(ox + 0.25f0, 0.25f0, 1f0) for (ox, _, _) in offsets]

"""Trace one ray per instance and return the result on the host."""
trace_rays_cpu(hwtlas, offsets) =
    tlastrace(TESTBACKEND, hwtlas, ray_origins(offsets), fill(Vec3f(0, 0, -1), length(offsets)))

# ------------------------------------------------------------------------------
# Correctness test
# ------------------------------------------------------------------------------

"""Assert that every ray hit matches the host reference for the current set of
`offsets` (one per instance): the unit triangle moved by its offset. Used after
each mutation."""
function assert_hits_match(r, offsets::Vector{NTuple{3,Float32}})
    @assert length(r.hit) == length(offsets)
    all_good = true
    for (i, (o, off)) in enumerate(zip(ray_origins(offsets), offsets))
        (want_hit, want_t, _) = nearesthit(o, Vec3f(0, 0, -1), [movedby(UNIT_TRI, off)])
        got_hit = r.hit[i]
        if got_hit != want_hit
            @warn "instance $i: hit mismatch" got_hit want_hit offset=off
            all_good = false
            continue
        end
        if want_hit && !isapprox(r.t[i], want_t; atol=1f-3)
            @warn "instance $i: t mismatch" got_t=r.t[i] want_t offset=off
            all_good = false
        end
    end
    return all_good
end

"""What the device's pool holds once the collector, a wait and a reclaim have given
back everything that was dropped: bytes taken from the device, regions on loan, and
blocks. The wait is what hands a collected array's region to the pool on a backend
that defers the destroy to its queue."""
function snapshot_state()
    dev = Mantle.Device(TESTBACKEND)
    p = Mantle.pool(dev)
    GC.gc(true); GC.gc(true)
    Mantle.waitidle(dev)
    Mantle.reclaim!(p, dev; wait = true)
    blocks = collect(Iterators.flatten(values(p.blocks)))
    (gpu_bytes = Mantle.reserved(p),
     live_regions = sum(b -> length(b.live), blocks; init = 0),
     pool_blocks = length(blocks))
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "HW HWTLAS — correctness under mutation" begin
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
            Raycore.update_transform!(hwtlas, handles[i], mat3x4translation(new_off...))
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
        h = push!(hwtlas, unit_triangle_mesh(), tlastranslation(off...);
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

@testset "HW HWTLAS — stress / leak bounds" begin
    N = 16
    hwtlas, handles, offsets = build_scene(N)

    # Warm up: do a couple of rebuild cycles + a trace, then GC. The
    # baseline we lock onto is the *post-warmup* state; the first few
    # iterations legitimately populate pools, kernel caches, etc.
    for _ in 1:4
        for i in 1:N
            offsets[i] = (Float32(2i), Float32(0.1), 0f0)
            Raycore.update_transform!(hwtlas, handles[i], mat3x4translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        _ = trace_rays_cpu(hwtlas, offsets)
    end
    baseline = snapshot_state()
    @info "baseline" baseline

    # Hammer the update cycle. Every iteration:
    #   * shuffle per-instance transforms
    #   * call sync!
    #   * trace + verify — catches use-after-free and UAF-masked noise
    n_iters   = 500
    max_hits  = 0  # debugging aid — peak samples during the loop
    for iter in 1:n_iters
        for i in 1:N
            offsets[i] = (Float32(2i + 0.01 * iter),
                          Float32(0.2 * cospi(iter / 7)),
                          Float32(-0.05 * sinpi(iter / 9)))
            Raycore.update_transform!(hwtlas, handles[i], mat3x4translation(offsets[i]...))
        end
        Raycore.sync!(hwtlas)
        hits = trace_rays_cpu(hwtlas, offsets)
        @assert assert_hits_match(hits, offsets) "iter $iter: hits diverged — UAF suspected"
        max_hits = max(max_hits, count(hits.hit))
    end
    final = snapshot_state()
    @info "after $n_iters iterations" final peak_hits=max_hits

    # Allow small growth (pool block fragmentation, kernel caches warming up),
    # but the delta must not scale with iteration count. These ceilings are
    # intentionally tight — anything looser stops being a real leak test.
    @testset "no unbounded GPU memory growth" begin
        @test final.gpu_bytes    <= baseline.gpu_bytes + 256 * 1024^2  # +256 MiB
        @test final.live_regions <= baseline.live_regions + 16
        @test final.pool_blocks  <= baseline.pool_blocks + 8
    end
end

# ------------------------------------------------------------------------------
# update_transforms! — bulk transform update
# ------------------------------------------------------------------------------

# "Refit, not rebuild" is read off the adapted form: a refit updates the
# structure in place, so `Adapt.adapt` keeps handing out the SAME object — which
# is what lets a caller (Hikari's integrator) cache it, and a recorded plan keep
# it, across transform updates. A rebuild hands out a new one.

@testset "HW HWTLAS — update_transforms! (CPU input) refits without rebuild" begin
    N = 8
    hwtlas, handles, offsets = build_scene(N)

    # Replace the per-handle batch with a single multi-instance handle so
    # we can exercise update_transforms! with a length>1 transforms vector.
    for h in handles
        Raycore.delete!(hwtlas, h)
    end
    init_xfs = [tlastranslation(Float32(2i), 0f0, 0f0) for i in 1:N]
    multi_handle = push!(hwtlas, unit_triangle_mesh(), init_xfs;
                         instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)
    pinned = Adapt.adapt(TESTBACKEND, hwtlas)

    # Drive update_transforms! 20 times; sync! must take the refit path
    # (same adapted form) every time.
    for iter in 1:20
        new_xfs = [mat3x4translation(Float32(2i + 0.05 * iter), 0f0, 0f0) for i in 1:N]
        Raycore.update_transforms!(hwtlas, multi_handle, new_xfs)
        Raycore.sync!(hwtlas)
        @test Adapt.adapt(TESTBACKEND, hwtlas) === pinned
    end

    # Final correctness check — rays at the post-update positions must hit, and
    # through the adapted form a caller kept from before the updates.
    final_offsets = [(Float32(2i + 0.05 * 20), 0f0, 0f0) for i in 1:N]
    @test assert_hits_match(trace_rays_cpu(pinned, final_offsets), final_offsets)
end

@testset "HW HWTLAS — update_transforms! accepts device-array input" begin
    N = 4
    init_xfs = [tlastranslation(Float32(2i), 0f0, 0f0) for i in 1:N]
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    h = push!(hwtlas, unit_triangle_mesh(), init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    new_xfs_cpu = [mat3x4translation(Float32(2i + 5f0), 0f0, 0f0) for i in 1:N]
    new_xfs_gpu = Mantle.devicearray(TESTBACKEND, new_xfs_cpu)
    Raycore.update_transforms!(hwtlas, h, new_xfs_gpu)
    Raycore.sync!(hwtlas)

    final_offsets = [(Float32(2i + 5f0), 0f0, 0f0) for i in 1:N]
    @test assert_hits_match(trace_rays_cpu(hwtlas, final_offsets), final_offsets)
end

@testset "HW HWTLAS — update_transforms! then delete!(handle) is safe" begin
    N = 4
    init_xfs = [tlastranslation(Float32(2i), 0f0, 0f0) for i in 1:N]
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    h_a = push!(hwtlas, unit_triangle_mesh(), init_xfs; instance_mask=UInt8(0xff))
    h_b = push!(hwtlas, unit_triangle_mesh(), tlastranslation(20f0, 0f0, 0f0))
    Raycore.sync!(hwtlas)

    new_xfs = [mat3x4translation(Float32(2i + 1f0), 0f0, 0f0) for i in 1:N]
    Raycore.update_transforms!(hwtlas, h_a, new_xfs)
    # Delete BEFORE syncing the refit -- topology change wins.
    @test Raycore.delete!(hwtlas, h_a) == true
    Raycore.sync!(hwtlas)
    @test Raycore.n_instances(hwtlas) == 1

    # Surviving batch still traces, and the deleted one is hit neither where it
    # was nor where the update would have put it.
    @test assert_hits_match(trace_rays_cpu(hwtlas, [(20f0, 0f0, 0f0)]), [(20f0, 0f0, 0f0)])
    gone = vcat([(Float32(2i), 0f0, 0f0) for i in 1:N], [(Float32(2i + 1f0), 0f0, 0f0) for i in 1:N])
    @test !any(trace_rays_cpu(hwtlas, gone).hit)
end

# ------------------------------------------------------------------------------
# Interleaved bulk update + trace at scale (refit path) — analogous to the
# Raycore SW stress, but the refit goes through the hardware structure's
# in-place update.
# Catches: per-frame staleness on the HW path, update-vs-build ordering bugs on
# the device, refit-vs-rebuild misclassification at high count.
# ------------------------------------------------------------------------------

@testset "HW HWTLAS — interleaved update_transforms! + trace tight loop (1000 inst, refit)" begin
    N = 1000
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    mesh = unit_triangle_mesh()
    init_xfs = [tlastranslation(Float32(2i), 0f0, 0f0) for i in 1:N]
    h = push!(hwtlas, mesh, init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    # Pin the adapted form — refit must keep it across all frames. If sync!
    # ever drops to rebuild for a frame (because it misread what changed) this
    # assertion trips.
    pinned = Adapt.adapt(TESTBACKEND, hwtlas)

    n_frames = 50
    for frame in 1:n_frames
        # Move every instance: x_i = 2i + 0.05*frame, y = small oscillation.
        new_offsets = [(Float32(2i + 0.05 * frame),
                        Float32(0.1 * sinpi(frame / 7)),
                        0f0)
                       for i in 1:N]
        new_xfs = [mat3x4translation(off...) for off in new_offsets]

        Raycore.update_transforms!(hwtlas, h, new_xfs)
        Raycore.sync!(hwtlas)
        @test Adapt.adapt(TESTBACKEND, hwtlas) === pinned    # refit, not rebuild

        # Trace every instance THIS frame; hit positions must reflect THIS
        # frame's transforms (not the previous frame's).
        hits = trace_rays_cpu(hwtlas, new_offsets)
        @test assert_hits_match(hits, new_offsets)
    end
end

# ------------------------------------------------------------------------------
# 500-iter mesh grow/shrink with HW raytracing every iter (correctness + leak)
# ------------------------------------------------------------------------------
#
# HW analogue of Raycore's grow/shrink test.  Each iter: drop the BLAS,
# build a fresh one with a new tessellation, sync (full rebuild, the old
# structure released once nothing in flight reads it), then trace through the
# new geometry. Catches:
#   * stale BLAS device-address captures the TLAS retains across rebuilds
#   * old instance / triangle / offset buffers being freed before the previous
#     iter's trace completed (UAF on the previous frame's reads)
#   * pool-block / live-region accumulation tied to per-iter mesh size churn

"""Analytic hit for a downward ray at (0, 0, 5) against a unit sphere at
(0, 0, offset_z): top of sphere at z=offset_z+1, t = 5 - offset_z - 1."""
sphere_hit_t(offset_z) = 5f0 - Float32(offset_z) - 1f0

# Sphere BLAS test scaffolding -- different geometry from unit_triangle_mesh,
# kept separate so the existing triangle-based tests aren't disturbed.
sphere_mesh_n(n::Int) = GeometryBasics.normal_mesh(
    GeometryBasics.Tessellation(GeometryBasics.Sphere(GeometryBasics.Point3f(0,0,0), 1f0), n))

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

@testset "HW HWTLAS — 500-iter mesh grow/shrink + HW trace per iter" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    h = push!(hwtlas, sphere_mesh_n(8), tlastranslation(0, 0, 0); instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)
    # Single ray (0, 0, 5) → -z direction.
    probe = TLASProbe(TESTBACKEND, [Point3f(0, 0, 5)], [Vec3f(0, 0, -1)])

    n_iters = 500
    saw_min, saw_max = typemax(Int), 0
    for iter in 1:n_iters
        tess  = hw_grow_shrink_tess(iter)
        saw_min, saw_max = min(saw_min, tess), max(saw_max, tess)
        z_off = Float32((iter % 30) * 0.05)            # 0 .. 1.45 — ray always reaches

        Raycore.delete!(hwtlas, h)
        h = push!(hwtlas, sphere_mesh_n(tess), tlastranslation(0, 0, z_off);
                  instance_mask=UInt8(0xff))
        Raycore.sync!(hwtlas)

        @test Raycore.n_instances(hwtlas) == 1

        # Trace through the freshly-built BVH.  If the previous iter's trace
        # hadn't completed before sync! freed the old structure and its arrays,
        # this trace would either fault or return wrong values — we'd see the t
        # mismatch immediately and not 500 iters from now.
        r = tlastrace(probe, hwtlas)
        @test r.hit[1]
        @test isapprox(r.t[1], sphere_hit_t(z_off); atol=0.15f0)
    end

    @test saw_min <= 10
    @test saw_max >= 90
end

@testset "HW HWTLAS — interleaved delete+push+sync+trace tight loop (500 inst, rebuild)" begin
    N = 500
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    mesh = unit_triangle_mesh()
    init_xfs = [tlastranslation(Float32(2i), 0f0, 0f0) for i in 1:N]
    h = push!(hwtlas, mesh, init_xfs; instance_mask=UInt8(0xff))
    Raycore.sync!(hwtlas)

    n_frames = 30
    for frame in 1:n_frames
        # Drop the whole batch and re-push — full topology rebuild per frame.
        Raycore.delete!(hwtlas, h)
        new_xfs = [tlastranslation(Float32(2i + 0.05 * frame),
                                Float32(0.1 * cospi(frame / 5)),
                                0f0)
                   for i in 1:N]
        new_offsets = [(Float32(2i + 0.05 * frame),
                        Float32(0.1 * cospi(frame / 5)),
                        0f0)
                       for i in 1:N]
        h = push!(hwtlas, mesh, new_xfs; instance_mask=UInt8(0xff))
        Raycore.sync!(hwtlas)

        # N and not 2N: the deleted batch is gone, not merely emptied.
        @test Raycore.n_instances(hwtlas) == N

        # Trace + verify against THIS frame's geometry.  A UAF on the freed
        # previous-frame instance or BLAS buffers would manifest as wrong hits
        # here (or a device fault, which the testset can't recover from).
        hits = trace_rays_cpu(hwtlas, new_offsets)
        @test assert_hits_match(hits, new_offsets)
    end
end

@testset "HW HWTLAS — n_instances matches live batch count under churn" begin
    # Batches of BLASes of different sizes pushed, deleted and moved in a random
    # order, with a `sync!` only every fifth step, so each sync sees several
    # mutations at once.
    rng = Random.MersenneTwister(0xCAFEBABE)
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handles = Raycore.TLASHandle[]
    sizes = Int[]
    expected = 0
    for iter in 1:80
        op = rand(rng, 1:3)
        if op == 1 && length(handles) < 12
            n = rand(rng, 1:5)
            xfs = [tlastranslation(Float32(2i + iter), 0f0, 0f0) for i in 1:n]
            h = push!(hwtlas, sphere_mesh_n(rand(rng, [4, 6, 8])), xfs)
            push!(handles, h)
            push!(sizes, n)
            expected += n
        elseif op == 2 && length(handles) > 0
            i = rand(rng, 1:length(handles))
            Raycore.delete!(hwtlas, handles[i])
            expected -= sizes[i]
            deleteat!(handles, i)
            deleteat!(sizes, i)
        elseif op == 3 && length(handles) > 0
            i = rand(rng, 1:length(handles))
            Raycore.update_transform!(hwtlas, handles[i],
                tlastranslation(Float32(rand(rng) * 6 - 3), 0f0, 0f0))
        end
        if iter % 5 == 0
            Raycore.sync!(hwtlas)
        end
        @test Raycore.n_instances(hwtlas) == expected
    end

    Raycore.sync!(hwtlas)
    @test Raycore.world_bound(hwtlas) isa Raycore.Bounds3
    @test Raycore.wait_for_gpu!(hwtlas) === hwtlas
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
