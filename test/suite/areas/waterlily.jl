# WaterLily on Mantle arrays: a consumer that brings its own
# KernelAbstractions kernels (`@loop` over CartesianIndices, launched on
# `get_backend(a)`), broadcasts, reductions on arrays and views (`sum`,
# `maximum`, `dot`), `similar`, `copy`, `fill!`, and a host read of the
# Poisson residual after every V-cycle. Mantle is only WaterLily's `mem`:
# `mem = data -> MantleArray(dev, data)` uploads each host array at creation
# (B3.18); every launch after that is eager on the device (api.md 2.2).
#
# The accuracy cases repeat WaterLily's own tests (test/test_flow.jl and
# test/helper.jl of WaterLily 1.8.0) with Mantle arrays. Phase 6: MantleArray
# as an AbstractGPUArray, the device as KA backend and the GPUArrays methods
# (mapreducedim! for the reductions) come there.

using WaterLily, StaticArrays, LinearAlgebra

"""WaterLily's `mem` for a Mantle device."""
onmantle(dev) = data -> MantleArray(dev, data)

"""The Taylor-Green vortex velocity (WaterLily test/helper.jl)."""
function taylorgreen(i, xy, t, κ, ν)
    x, y = @. xy * κ
    i == 1 && return -sin(x) * cos(y) * exp(-2κ^2 * ν * t)
    return cos(x) * sin(y) * exp(-2κ^2 * ν * t)
end

"""WaterLily's periodic Taylor-Green simulation, L = 64 (test/helper.jl `TGVsim`)."""
function taylorgreensim(mem; T = Float32, Re = 1e8)
    L = 64
    κ = T(2π / L)
    ν = T(1 / (κ * Re))
    return Simulation((L, L), (i, x, t) -> taylorgreen(i, x, t, κ, ν), L; U = 1, ν, T, mem, perdir = (1, 2))
end

"""WaterLily's circle in an accelerating flow (test/test_flow.jl): radius 32 in a (1024, 1024) domain."""
acceleratingcircle(mem; radius = 32, H = 16) =
    Simulation(radius .* (2H, 2H), (i, x, t) -> i == 1 ? t : zero(t), radius;
               U = 1, mem, body = AutoBody((x, t) -> √sum(abs2, x .- H * radius) - radius))

"""A static sphere of radius L/4 in a (2L, L, L) domain with uniform inflow."""
spheresim(mem; L = 32, Re = 250) =
    Simulation((2L, L, L), (1, 0, 0), L; ν = L / Re, T = Float32, mem,
               body = AutoBody((x, t) -> √sum(abs2, x .- SA[L ÷ 2, L ÷ 2, L ÷ 2]) - L ÷ 4))

"""`n` time steps of a simulation whose body does not move."""
simsteps!(sim, n) = foreach(_ -> sim_step!(sim; remeasure = false), 1:n)

"""A sphere simulation created, stepped twice and dropped: kernels compiled, the pool grown to its need."""
warmsphere(dev) = (simsteps!(spheresim(onmantle(dev)), 2); nothing)

reldiff(a, b) = norm(a .- b) / norm(b)

# WL-01: the Taylor-Green vortex after sim_step!(sim, π/100) matches the
# analytic solution to 1e-4 in L2, as WaterLily's "Flow.jl periodic TGV"
# test, with every field a MantleArray on the test device.
@case "WL-01" 6 begin
    dev = testdevice()
    sim = taylorgreensim(onmantle(dev))
    @test sim.flow.u isa MantleArray
    @test sim.pois.levels[end].x isa MantleArray     # the multigrid levels come from `similar`
    @test get_backend(sim.flow.u) === dev
    ue = Array(copy(sim.flow.u))
    sim_step!(sim, π / 100)
    apply!((i, x) -> taylorgreen(i, x, WaterLily.time(sim), 2π / sim.L, sim.flow.ν), ue)
    u = Array(sim.flow.u)
    @test WaterLily.L₂(u[:, :, 1] .- ue[:, :, 1]) < 1e-4
    @test WaterLily.L₂(u[:, :, 2] .- ue[:, :, 2]) < 1e-4
end

# WL-02: the accelerating circle's added-mass force is ≈ [-1, 0] within 0.04
# and the flow peaks near 2U, as WaterLily's "Circle in accelerating flow"
# test; after three more steps the Poisson solver needs at most 2 V-cycles.
# `pressure_force` sums in Float64 on the device (Metrics.jl: `sum(Float64,
# df; dims)`).
@case "WL-02" 6 begin
    dev = testdevice()
    sim = acceleratingcircle(onmantle(dev))
    sim_step!(sim)
    @test isapprox(WaterLily.pressure_force(sim) / (π * sim.L^2), [-1, 0]; atol = 0.04)
    u = Array(sim.flow.u)
    @test maximum(u) / u[2, 2, 1] > 1.91
    foreach(_ -> sim_step!(sim), 1:3)
    @test all(sim.pois.n .≤ 2)
end

# WL-03: a static sphere run for 10 steps on the device equals the same run
# with `mem = Array` within relative 1e-4, with the same Poisson iteration
# counts (the residual read back after every V-cycle decides them) and the
# same time steps (the CFL `maximum` read back after every step).
@case "WL-03" 6 begin
    dev = testdevice()
    sim, ref = spheresim(onmantle(dev)), spheresim(Array)
    simsteps!(sim, 10)
    simsteps!(ref, 10)
    @test reldiff(Array(sim.flow.u), ref.flow.u) < 1e-4
    @test reldiff(Array(sim.flow.p), ref.flow.p) < 1e-4
    @test sim.pois.n == ref.pois.n
    @test sim.flow.Δt ≈ ref.flow.Δt rtol = 1e-4
end

# WL-04: time per step per degree of freedom (cells of the domain, ghost cells
# excluded, as WaterLily's benchmarks count them) of the sphere at L = 64.
# Recorded, not gated. Every step ends with a host read (the CFL maximum), so
# the elapsed time covers the GPU work.
@case "WL-04" 6 :long begin
    dev = testdevice()
    L, steps = 64, 20
    sim = spheresim(onmantle(dev); L)
    simsteps!(sim, 2)
    t = @elapsed simsteps!(sim, steps)
    nsperdof = 1e9 * t / (prod(size(sim.flow.p) .- 2) * steps)
    @info "WL-04: WaterLily sphere, ns per DOF per step" device = devicename(dev) L steps nsperdof
    @test nsperdof > 0
end

# WL-05: in steady state a step compiles no kernel (every `@loop` kernel comes
# from the device type's cache) and no plan (eager work builds none); the
# temporaries of WaterLily's reductions and the simulation's arrays are given
# back once the simulation is dropped.
@case "WL-05" 6 begin
    dev = testdevice()
    warmsphere(dev)
    before = memory(dev)
    sim = spheresim(onmantle(dev))
    simsteps!(sim, 1)
    _, d = counted(() -> simsteps!(sim, 1))
    @test d.kernelcompiles == 0
    @test d.plancompiles == 0
    sim = nothing
    after = memory(dev)
    @test after.uploads == before.uploads
    @test after.cells == before.cells
    @test after.reserved <= before.reserved
    @test after.mapped <= before.mapped
end
