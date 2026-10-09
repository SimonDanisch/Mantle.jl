"""
A kernel that traces an HWTLAS can be launched AND recorded.

Every owner of a submission holds what its kernels read, by walking each
argument's fields until it finds an array (`holdleaves!`). The adapted structure
a kernel is handed carries the HWTLAS itself, for the backend to bind, and on
Vulkan that is a cycle: a `VulkanTLAS` holds its submission channel, the channel
holds the context, the context holds the channel. It is also unnecessary — the
arrays a ray query walks are exposed directly as fields of the `AdaptedAccel` —
so there is a method that stops the walk there.

That method was written for a single closed-command-buffer owner, from when a
one-shot was the only thing a claim could go into. A `Recording` is the other
one, so a hardware-RT plan fell through to the generic walker the first time it
was RECORDED rather than launched, and the walk reached `VK.Instance` — whose
`destructor` field is a closure over the instance itself. 53 320 frames and a
`StackOverflowError`, from Hikari's hardware trace pass.

Both owners are exercised the way a caller reaches them: an eager launch (a
one-shot) and a plan that is recorded and run. The overflow corrupts the session
it happens in, so a regression here takes the rest of the run with it — which is
loud, and the reason this file is small.
"""

using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Mantle, Adapt, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "a kernel tracing an HWTLAS launches, and its plan records and runs" begin
    dev = Mantle.Device(TESTBACKEND)
    hwtlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(hwtlas, GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0), 1f0)),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(1))
    Raycore.sync!(hwtlas)
    accel = Adapt.adapt(TESTBACKEND, hwtlas)

    origins = [Point3f(0, 0, 5), Point3f(0.2f0, 0, 5), Point3f(5, 5, 5)]
    dirs = fill(Vec3f(0, 0, -1), 3)
    want = [true, true, false]

    # The one-shot owner: an eager launch.
    r = tlastrace(TESTBACKEND, accel, origins, dirs)
    @test r.hit == want
    @test r.id == UInt32[1, 1, 0]

    # The recording owner: the same kernel as a pass of a plan. `record!` is
    # where the walk overflowed.
    n = length(origins)
    hits = Mantle.Buffer(dev, zeros(Int32, n))
    ts = Mantle.Buffer(dev, zeros(Float32, n))
    ids = Mantle.Buffer(dev, zeros(UInt32, n))
    cents = Mantle.Buffer(dev, zeros(Point3f, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, tlasprobe!, (hits, ts, ids, cents, Mantle.Buffer(dev, origins),
                                     Mantle.Buffer(dev, dirs), accel), n; name = "trace")
    pl = Mantle.Plan(g)
    Mantle.record!(pl)
    @test Mantle.recorded(pl) == Mantle.recordable(dev, pl)
    Mantle.run!(pl)
    KernelAbstractions.synchronize(TESTBACKEND)
    @test Array(Mantle.storage(hits)) == Int32.(want)
    @test Array(Mantle.storage(ids)) == UInt32[1, 1, 0]
    @test isapprox(Array(Mantle.storage(ts))[1], 4f0; atol = 0.05f0)
    Mantle.free!(pl)
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
