using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava

# ===============================================================================
# Phase-C contract: `sync!(hwtlas)` removes the two unconditional
# `KA.synchronize(hwtlas.backend)` calls that the old Raycore.HWTLAS had.
# What it does NOT remove is the fence wait inside `Mantle.as_build` — the
# Vulkan AS-build step necessarily waits for its own build command to
# complete (reads vertex/index data).  Queue-FIFO semantics mean that
# wait implicitly drains any prior work on the same queue, but that is
# scoped + narrow, not a backend-wide sync.
#
# Tested here:
#
#   1. Source-level: `hwtlas.jl`'s `sync!` must not call `KA.synchronize`
#      (the symbol that Phase C removed).  This is the direct regression
#      guard — if a future refactor puts the call back, this test trips
#      immediately regardless of what the GPU happens to be doing.
#
#   2. Behavioural: `sync!` on a dirty topology change, *with no prior
#      in-flight GPU work*, returns well under the wall-clock time that a
#      full-backend synchronize would take even in the degenerate "GPU
#      is idle" case.  An upper bound of 500 ms is a loose regression
#      guard — real `sync!` CPU cost on a 128-tessellation sphere is
#      well under 10 ms; anything approaching half a second implies an
#      unexpected blocking primitive was introduced.
# ===============================================================================

@testset "Mantle.HWTLAS — sync! contains no KA.synchronize" begin
    path = joinpath(dirname(pathof(Lava)), "raytracing", "hwtlas.jl")
    src  = read(path, String)
    # Extract the body of `sync!` so comments in other functions don't fool us.
    m = match(r"function Raycore\.sync!\(hwtlas::HWTLAS\)(.*?)\nend"s, src)
    @test m !== nothing
    sync_body = m === nothing ? "" : m.captures[1]
    # Strip line comments so a future "# we used to call KA.synchronize here"
    # comment doesn't trip this test.
    sync_body_code = replace(sync_body, r"#[^\n]*" => "")
    @test !occursin("KA.synchronize", sync_body_code)
    @test !occursin("wait_on_timeline", sync_body_code)
end

@testset "Mantle.HWTLAS — sync! CPU time is bounded on idle queue" begin
    backend = Mantle.LavaBackend()
    hwtlas = Mantle.HWTLAS(backend)
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
    # rather than against the specific KA.synchronize call (caught by
    # the static test above).
    @test t_sync < 0.5
end
