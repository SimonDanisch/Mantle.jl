# ==============================================================================
# HW HWTLAS mesh-update tests
# ==============================================================================
#
# These tests were split off from Raycore's test_mesh_update.jl in Phase F of
# the VulkanTLAS release cleanup. They exercise the hardware HWTLAS, on whichever
# backend builds one: correctness under size-oscillating mesh swaps, the
# adapted-form ownership contract of `sync!`, transform propagation via `sync!`,
# per-instance data surviving a refit, and a GPU-resource leak bound. Every
# result is read off a trace (`test/hwtlas_helpers.jl`).
# ==============================================================================

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using Mantle
using Adapt
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const Tri = Raycore.Triangle{UInt32}

"""Unit sphere centred at origin; `n` = tessellation count."""
function sphere_mesh(n::Int)
    GeometryBasics.normal_mesh(Tessellation(Sphere(Point3f(0), 1f0), n))
end

"""Ray straight down the +z axis from z=5. For a unit sphere translated by
`offset_z`, the closest hit is at t = 4 - offset_z."""
expected_t(offset_z::Real) = Float32(5) - Float32(offset_z) - Float32(1)

"""The one ray every test here traces: down the +z axis from z=5. Built once per
testset and reused, so allocation noise does not mask the leak test."""
axisprobe() = TLASProbe(TESTBACKEND, [Point3f(0, 0, 5)], [Vec3f(0, 0, -1)])

"""Trace one ray down the +z axis and return (hit, t) on the host."""
function hw_trace_one(probe, hwtlas)
    r = tlastrace(probe, hwtlas)
    return (hit = r.hit[1], t = r.t[1])
end

"""Swap the mesh in `hwtlas` to a fresh sphere at `offset_z` with tessellation `n`."""
function hw_swap_mesh!(hwtlas, handle, n, offset_z)
    Raycore.delete!(hwtlas, handle)
    new_handle = push!(hwtlas, sphere_mesh(n), tlastranslation(0, 0, offset_z))
    Raycore.sync!(hwtlas)
    return new_handle
end

"""Snapshot what the device's pool holds, once the collector and a reclaim have
given back everything that was dropped: bytes taken from the device, regions on
loan, and blocks."""
function snapshot_state_hw()
    dev = Mantle.Device(TESTBACKEND)
    p = Mantle.pool(dev)
    GC.gc(true); GC.gc(true)
    Mantle.reclaim!(p, dev; wait = true)
    blocks = collect(Iterators.flatten(values(p.blocks)))
    return (gpu_bytes = Mantle.reserved(p),
            live_regions = sum(b -> length(b.live), blocks; init = 0),
            pool_blocks = length(blocks))
end

# ------------------------------------------------------------------------------

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "HW HWTLAS — mesh update correctness under size oscillation" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    probe = axisprobe()
    handle = push!(hwtlas, sphere_mesh(16), tlastranslation(0, 0, 0))
    Raycore.sync!(hwtlas)

    # Baseline
    r = hw_trace_one(probe, hwtlas)
    @test r.hit
    @test isapprox(r.t, expected_t(0); atol=0.05f0)

    # Oscillate small -> big -> small -> bigger -> smaller.
    tess_schedule = [32, 8, 48, 12, 64, 16, 8, 32, 96, 16]
    for (i, n) in enumerate(tess_schedule)
        offset_z = Float32(0.05 * i)
        handle = hw_swap_mesh!(hwtlas, handle, n, offset_z)
        r = hw_trace_one(probe, hwtlas)
        @test r.hit         broken=false
        @test isapprox(r.t, expected_t(offset_z); atol=0.1f0)
    end
end

@testset "HW HWTLAS — adapt-once-then-mutate via the adapted form" begin
    # Invariant: sync!(hwtlas) is the single owner of the adapted form, and MAY
    # replace it when a buffer was reallocated. `AdaptedAccel` holds the adapted
    # device buffers, not a reference to the mutable HWTLAS, so whether the
    # wrapper survives a rebuild depends on whether every buffer could be reused
    # — and the acceleration-structure sizes are the driver's to report. This
    # asserted `===` across the rebuild, which held on RADV and not on NVIDIA;
    # failing, it then printed the stale wrapper, whose buffers `sync!` had freed,
    # and errored.
    #
    # What a consumer may rely on, and what is asserted: with nothing changed,
    # `adapt` hands out the same adapted form again, and after `sync!` tracing
    # sees the mutation.
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    probe = axisprobe()
    handle = push!(hwtlas, sphere_mesh(16), tlastranslation(0, 0, 0))

    st_before = Adapt.adapt(TESTBACKEND, hwtlas)
    @test Adapt.adapt(TESTBACKEND, hwtlas) === st_before

    r_before = hw_trace_one(probe, st_before)
    @test r_before.hit
    @test isapprox(r_before.t, expected_t(0); atol=0.05f0)

    # Mutate: swap sphere to z=2 — expected t moves 4.0 -> 2.0.
    Raycore.delete!(hwtlas, handle)
    handle = push!(hwtlas, sphere_mesh(48), tlastranslation(0, 0, 2f0))
    Raycore.sync!(hwtlas)

    st_after = Adapt.adapt(TESTBACKEND, hwtlas)
    @test Adapt.adapt(TESTBACKEND, hwtlas) === st_after

    r_after = hw_trace_one(probe, st_after)
    @test r_after.hit
    @test isapprox(r_after.t, expected_t(2); atol=0.1f0)
end

@testset "HW HWTLAS — transform update via sync!(hwtlas)" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    probe = axisprobe()
    handle = push!(hwtlas, sphere_mesh(16), tlastranslation(0, 0, 0))
    Raycore.sync!(hwtlas)

    r1 = hw_trace_one(probe, hwtlas)
    @test r1.hit
    @test isapprox(r1.t, expected_t(0); atol=0.05f0)

    # Move instance to z=1.5.
    Raycore.update_transform!(hwtlas, handle, tlastranslation(0, 0, 1.5f0))
    Raycore.sync!(hwtlas)

    r2 = hw_trace_one(probe, hwtlas)
    @test r2.hit
    @test isapprox(r2.t, expected_t(1.5); atol=0.1f0)
end

# One transform for every instance of a batch, the meaning `update_transform!` has
# on every structure, Raycore's own included. Metal once refused a batch of more
# than one, and Vulkan answered an unknown handle with `false` where the others
# threw.
@testset "HW HWTLAS — update_transform! moves every instance of a batch" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle = push!(hwtlas, sphere_mesh(16), [tlastranslation(-3, 0, 0), tlastranslation(3, 0, 0)])
    Raycore.sync!(hwtlas)
    # Down -z at x = -3, x = 3 and x = 10.
    probe = TLASProbe(TESTBACKEND, [Point3f(-3, 0, 5), Point3f(3, 0, 5), Point3f(10, 0, 5)],
                      fill(Vec3f(0, 0, -1), 3))
    @test tlastrace(probe, hwtlas).hit == [true, true, false]

    Raycore.update_transform!(hwtlas, handle, tlastranslation(10, 0, 0))
    Raycore.sync!(hwtlas)
    @test Raycore.n_instances(hwtlas) == 2
    @test tlastrace(probe, hwtlas).hit == [false, false, true]

    @test_throws ArgumentError Raycore.update_transform!(
        hwtlas, Raycore.TLASHandle(typemax(UInt32)), tlastranslation(0, 0, 0))
end

# A refit must not flatten what `push!` wrote into each instance.
#
# Vulkan's `update_instance_records_kernel!` once REBUILT every record from
# scalars taken off the batch: one `custom_index`, one mask, one SBT offset, and
# zero flags. But `push!(hwtlas, mesh, transform; instance_id)` puts the id in the
# RECORD, and the CPU-record `addbatch!` passed `UInt32(0)` for the batch's
# field — so the first `update_transform!` reset `gl_InstanceCustomIndexEXT` to
# 0 with no error, and a shader keyed on it silently read instance 0's
# material. The vector form could not work even in principle: `instance_ids` is
# per instance and a batch field is one value.
#
# Traced: the probe returns the custom index, which is the field a shader reads,
# and a mask a refit zeroed would make its instance unhittable. The SBT offset
# only selects a hit group in a ray-tracing pipeline's shader binding table,
# which a ray query never consults, so a trace cannot see it.
@testset "HW HWTLAS — a refit preserves per-instance ids and masks" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    ids = UInt32[7, 11, 23]
    xs = Float32[0, 3, 6]
    handle = push!(hwtlas, sphere_mesh(16),
                   [tlastranslation(x, 0, 0) for x in xs];
                   instance_ids = ids, instance_mask = UInt8(0x0f),
                   sbt_offset = UInt32(2))
    # …and beside it an instance no ray hits: mask 0 matches no ray's cull mask.
    push!(hwtlas, sphere_mesh(16), tlastranslation(9f0, 0f0, 0f0);
          instance_id = UInt32(99), instance_mask = UInt8(0x00))
    Raycore.sync!(hwtlas)

    probe = TLASProbe(TESTBACKEND, [Point3f(x, 0, 5) for x in [xs; 9f0]],
                      fill(Vec3f(0, 0, -1), 4))
    before = tlastrace(probe, hwtlas)
    @test before.hit == [true, true, true, false]
    @test before.id[1:3] == ids                  # what push! wrote
    @test all(t -> isapprox(t, expected_t(0); atol = 0.05f0), before.t[1:3])

    # A bulk transform update, which is what goes through the refit.
    Raycore.update_transforms!(hwtlas, handle,
        Mantle.devicearray(TESTBACKEND,
            [Mantle.mat4_to_vk_transform(tlastranslation(x, 0f0, 1f0)) for x in xs]))
    Raycore.sync!(hwtlas)

    after = tlastrace(probe, hwtlas)
    @test after.hit == [true, true, true, false] # the masks survived, the 0 one too
    @test after.id[1:3] == ids                   # the ids survived: was [0, 0, 0]
    @test all(t -> isapprox(t, expected_t(1); atol = 0.05f0), after.t[1:3])  # the transform moved
end

# Not ported: "hw_accel + rt_pipeline reused across sync! rebuilds" asserted that
# Vulkan's `HardwareAccel` and the shader binding table compiled into it are the
# same objects after a rebuild — a compile-once property of one backend's
# ray-tracing pipeline with nothing a trace can observe.

@testset "HW HWTLAS — mesh update leak bound (GPU resources)" begin
    hwtlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    probe = axisprobe()
    handle = push!(hwtlas, sphere_mesh(16), tlastranslation(0, 0, 0))
    Raycore.sync!(hwtlas)

    # Warm up: a few cycles to settle pool/kernel caches.
    for _ in 1:4
        handle = hw_swap_mesh!(hwtlas, handle, 32, 0.1f0)
        _ = hw_trace_one(probe, hwtlas)
    end
    baseline = snapshot_state_hw()
    @info "HW HWTLAS leak test baseline" baseline

    n_iters = 100
    for iter in 1:n_iters
        n = iseven(iter) ? 16 : 48
        offset_z = Float32(0.01 * iter)
        handle = hw_swap_mesh!(hwtlas, handle, n, offset_z)
        r = hw_trace_one(probe, hwtlas)
        @assert r.hit "iter $iter: ray missed — possible UAF or stale BLAS"
        @assert isapprox(r.t, expected_t(offset_z); atol=0.15f0) "iter $iter: wrong t=$(r.t), expected $(expected_t(offset_z))"
    end
    final = snapshot_state_hw()
    @info "HW HWTLAS leak test after $n_iters iters" final

    @testset "no unbounded GPU memory growth" begin
        @test final.gpu_bytes    <= baseline.gpu_bytes + 256 * 1024^2    # +256 MiB
        @test final.live_regions <= baseline.live_regions + 16
        @test final.pool_blocks  <= baseline.pool_blocks + 8
    end
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
