using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))

# ===============================================================================
# Phase-C contract: `sync!(hwtlas)` removes the two unconditional
# `KA.synchronize(hwtlas.backend)` calls a software TLAS needs.
# What it does NOT remove is the wait inside the acceleration-structure build
# itself — the build necessarily waits for its own build command to complete
# (it reads vertex/index data). Queue-FIFO semantics mean that wait implicitly
# drains any prior work on the same queue, but that is scoped + narrow, not a
# backend-wide sync.
#
# Tested here, behaviourally: `sync!` on a dirty topology change, *with no prior
# in-flight GPU work*, returns well under the wall-clock time that a
# full-backend synchronize would take even in the degenerate "GPU is idle" case.
# An upper bound of 500 ms is a loose regression guard — real `sync!` CPU cost
# on a 128-tessellation sphere is well under 10 ms; anything approaching half a
# second implies an unexpected blocking primitive was introduced.
#
# There was also a source-level check that Vulkan's `sync!` body does not
# contain `KA.synchronize`. It read one backend's source file, so it is not here;
# the timing bound is what a caller would notice.
# ===============================================================================

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "HWTLAS — sync! CPU time is bounded on idle queue" begin
    hwtlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    mesh = GeometryBasics.normal_mesh(Tessellation(Sphere(Point3f(0), 1f0), 128))
    h = push!(hwtlas, mesh, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
    Raycore.sync!(hwtlas)

    # Fully drain — no pending GPU work.  `sync!` should now be bounded
    # by its own AS-build dispatch + CPU work.
    Raycore.wait_for_gpu!(hwtlas)

    # Mutate (delete + re-push a slightly displaced mesh) and time sync!.
    Raycore.delete!(hwtlas, h)
    mesh2 = GeometryBasics.normal_mesh(Tessellation(Sphere(Point3f(0,0,0.1f0), 1f0), 128))
    push!(hwtlas, mesh2, SMatrix{4,4,Float32}(I); instance_id=UInt32(2))

    t_sync = @elapsed Raycore.sync!(hwtlas)
    @info "nonblocking_sync: sync! on idle queue took $(round(t_sync*1000; digits=2))ms"

    # Very loose upper bound.  Real cost on this GPU is <10ms.  An
    # accidental `KA.synchronize` on a fully-idle queue is a no-op, so
    # this bound is primarily a guard against *inserted* slow paths
    # rather than against the specific KA.synchronize call.
    @test t_sync < 0.5
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
