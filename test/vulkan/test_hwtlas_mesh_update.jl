# ==============================================================================
# HW HWTLAS mesh-update tests (Lava backend)
# ==============================================================================
#
# These tests were split off from Raycore's test_mesh_update.jl in Phase F of
# the VulkanTLAS release cleanup. They exercise MVE.VulkanTLAS specifically:
# correctness under size-oscillating mesh swaps, the static_tlas ownership
# contract, transform propagation via sync!, rt_pipeline identity preservation,
# and a GPU-resource leak bound.
# ==============================================================================

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using Lava, Mantle
using Adapt

const HW_BACKEND = MVE.LavaBackend()

# Reuse buffers across iterations so allocation noise doesn't mask the leak test.
const HW_RAYS_BUF = MVE.LavaArray([Raycore.RTRay(0f0, 0f0, 5f0, 0f0, 0f0, 0f0, -1f0, 1f3)])
const HW_HITS_BUF = MVE.LavaArray(fill(Raycore.RTHitResult(0, 0, 0, 0, 0, 0, 0, 0), 1))

"""Unit sphere centred at origin; `n` = tessellation count."""
function sphere_mesh(n::Int)
    GeometryBasics.normal_mesh(Tessellation(Sphere(Point3f(0), 1f0), n))
end

"""Ray straight down the +z axis from z=5. For a unit sphere translated by
`offset_z`, the closest hit is at t = 4 - offset_z."""
expected_t(offset_z::Real) = Float32(5) - Float32(offset_z) - Float32(1)

# Shared HW-HWTLAS helpers — see test/hwtlas_helpers.jl (provides `translation`).
isdefined(@__MODULE__, :translation) ||
    include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

"""Trace one ray down the +z axis via HW RT and return (hit, t) on CPU."""
function hw_trace_one(hwtlas)
    Mantle.trace_closest_hits!(HW_HITS_BUF, HW_RAYS_BUF, hwtlas.hw_accel, 1)
    Raycore.wait_for_gpu!(hwtlas)
    r = Array(HW_HITS_BUF)[1]
    return (hit = r.hit != UInt32(0), t = Float32(r.t))
end

"""Swap the mesh in `hwtlas` to a fresh sphere at `offset_z` with tessellation `n`."""
function hw_swap_mesh!(hwtlas, handle, n, offset_z)
    Raycore.delete!(hwtlas, handle)
    new_handle = push!(hwtlas, sphere_mesh(n), translation(0, 0, offset_z))
    Raycore.sync!(hwtlas)
    return new_handle
end

"""Snapshot GPU resource counters."""
function snapshot_state_hw()
    gpu_bytes  = MVE.gpu_live_bytes()
    live_bufs  = MVE.live_buffer_count()
    pool_blocks = length(MVE.poolblocks(MVE.vk_context()))
    return (gpu_bytes=gpu_bytes, live_bufs=live_bufs, pool_blocks=pool_blocks)
end

# ------------------------------------------------------------------------------

@testset "HW HWTLAS — mesh update correctness under size oscillation" begin
    hwtlas = MVE.VulkanTLAS(HW_BACKEND)
    handle = push!(hwtlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(hwtlas)

    # Baseline
    r = hw_trace_one(hwtlas)
    @test r.hit
    @test isapprox(r.t, expected_t(0); atol=0.05f0)

    # Oscillate small -> big -> small -> bigger -> smaller.
    tess_schedule = [32, 8, 48, 12, 64, 16, 8, 32, 96, 16]
    for (i, n) in enumerate(tess_schedule)
        offset_z = Float32(0.05 * i)
        handle = hw_swap_mesh!(hwtlas, handle, n, offset_z)
        r = hw_trace_one(hwtlas)
        @test r.hit         broken=false
        @test isapprox(r.t, expected_t(offset_z); atol=0.1f0)
    end
end

@testset "HW HWTLAS — adapt-once-then-mutate via hwtlas.static_tlas" begin
    # Invariant: sync!(hwtlas) is the single owner of hwtlas.static_tlas. A
    # consumer that holds hwtlas.static_tlas (a thin AdaptedAccel wrapper around
    # the mutable VulkanTLAS) sees any mutation that went through push!/delete! + sync!
    # because the wrapper always references the live mutable struct.
    #
    # Note: unlike StaticTLAS (which is a value-snapshot), AdaptedAccel is a
    # thin immutable wrapper holding a mutable VulkanTLAS reference. Two wrappers around
    # the same VulkanTLAS are always ===. The identity-change contract doesn't apply here
    # — instead we verify that trace results reflect the mutation.
    hwtlas = MVE.VulkanTLAS(HW_BACKEND)
    handle = push!(hwtlas, sphere_mesh(16), translation(0, 0, 0))

    st_before = Adapt.adapt(HW_BACKEND, hwtlas)
    @test hwtlas.static_tlas === st_before

    r_before = hw_trace_one(hwtlas)
    @test r_before.hit
    @test isapprox(r_before.t, expected_t(0); atol=0.05f0)

    # Mutate: swap sphere to z=2 — expected t moves 4.0 -> 2.0.
    Raycore.delete!(hwtlas, handle)
    handle = push!(hwtlas, sphere_mesh(48), translation(0, 0, 2f0))
    Raycore.sync!(hwtlas)

    # AdaptedAccel wraps the live mutable VulkanTLAS — the wrapper identity is
    # stable (=== holds) but the underlying geometry has changed.
    st_after = hwtlas.static_tlas
    @test st_after === st_before    # same thin wrapper, updated internals

    r_after = hw_trace_one(hwtlas)
    @test r_after.hit
    @test isapprox(r_after.t, expected_t(2); atol=0.1f0)
end

@testset "HW HWTLAS — transform update via sync!(hwtlas)" begin
    hwtlas = MVE.VulkanTLAS(HW_BACKEND)
    handle = push!(hwtlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(hwtlas)

    r1 = hw_trace_one(hwtlas)
    @test r1.hit
    @test isapprox(r1.t, expected_t(0); atol=0.05f0)

    # Move instance to z=1.5.
    Raycore.update_transform!(hwtlas, handle, translation(0, 0, 1.5f0))
    Raycore.sync!(hwtlas)

    r2 = hw_trace_one(hwtlas)
    @test r2.hit
    @test isapprox(r2.t, expected_t(1.5); atol=0.1f0)
end

@testset "HW HWTLAS — hw_accel + rt_pipeline reused across sync! rebuilds" begin
    # The HardwareAccel (and thus the RT pipeline compiled into the SBT) must
    # survive mesh swaps — one RT pipeline per VulkanTLAS, not per rebuild.
    hwtlas = MVE.VulkanTLAS(HW_BACKEND)
    handle = push!(hwtlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(hwtlas)

    accel_before = hwtlas.hw_accel
    pipeline_before = accel_before.rt_pipeline

    # Swap mesh (topology change forces a full rebuild).
    handle = hw_swap_mesh!(hwtlas, handle, 32, 0f0)

    accel_after = hwtlas.hw_accel
    @test accel_after === accel_before          # same HardwareAccel object reused
    @test accel_after.rt_pipeline === pipeline_before   # same compiled RT pipeline
end

@testset "HW HWTLAS — mesh update leak bound (GPU resources)" begin
    hwtlas = MVE.VulkanTLAS(HW_BACKEND)
    handle = push!(hwtlas, sphere_mesh(16), translation(0, 0, 0))
    Raycore.sync!(hwtlas)

    # Warm up: a few cycles to settle pool/kernel caches.
    for _ in 1:4
        handle = hw_swap_mesh!(hwtlas, handle, 32, 0.1f0)
        _ = hw_trace_one(hwtlas)
    end
    GC.gc(true); GC.gc(true)
    baseline = snapshot_state_hw()
    @info "HW HWTLAS leak test baseline" baseline

    n_iters = 100
    for iter in 1:n_iters
        n = iseven(iter) ? 16 : 48
        offset_z = Float32(0.01 * iter)
        handle = hw_swap_mesh!(hwtlas, handle, n, offset_z)
        r = hw_trace_one(hwtlas)
        @assert r.hit "iter $iter: ray missed — possible UAF or stale BLAS"
        @assert isapprox(r.t, expected_t(offset_z); atol=0.15f0) "iter $iter: wrong t=$(r.t), expected $(expected_t(offset_z))"
    end
    GC.gc(true); GC.gc(true)
    final = snapshot_state_hw()
    @info "HW HWTLAS leak test after $n_iters iters" final

    @testset "no unbounded GPU memory growth" begin
        @test final.gpu_bytes   <= baseline.gpu_bytes + 256 * 1024^2    # +256 MiB
        @test final.live_bufs   <= baseline.live_bufs + 16
        @test final.pool_blocks <= baseline.pool_blocks + 8
    end
end

println("\nAll HW HWTLAS mesh-update tests passed.")
