"""
The Metal hardware acceleration structure and the traversal a Julia kernel does
against it.

This is the path `hw_accel=true` takes. It exists because a Julia kernel CAN
call hardware traversal on Metal, contrary to the obvious reading: the
`intersector<>` is a C++ template with no AIR symbol, but a `[[visible]]` MSL
function wrapping it can be LINKED into the kernel's pipeline
(`MTLLinkedFunctions`), and the acceleration structure reaches it through an
argument buffer because a Julia kernel cannot declare an
`instance_acceleration_structure` parameter.

Three things in that chain fail silently rather than loudly, and each has an
assertion below:

  * residency — a TLAS bound through an argument buffer is not automatically
    resident, and a non-resident structure reports every ray as a miss;
  * the 24-byte hit record is an ABI shared with hand-written MSL that no
    compiler checks;
  * the fifth return value is the instance CUSTOM index, not Metal's
    `instance_id`. Returning the latter type-checks, runs, and quietly gives
    every instance past the first the wrong medium interface.
"""

using Test, Mantle, Metal, Raycore, GeometryBasics, KernelAbstractions
using Adapt, StaticArrays
# QUALIFIED below, not `using LinearAlgebra`: these files are included into
# `Main` one after another, and once some earlier one has touched an undefined
# `Main.normalize` the binding is resolved and a later `using` cannot introduce
# it — an `UndefVarError` whose cause is the ORDER of the includes.
import LinearAlgebra
const KA = KernelAbstractions
const MEXT = Base.get_extension(Mantle, :MantleMetalExt)

@kernel function _hw_probe!(hits, ts, insts, dirs, accel, O)
    i = @index(Global)
    hit, tri, t, bary, inst = Raycore.closest_hit(accel, Raycore.Ray(o = O, d = dirs[i], t_max = 1f30))
    hits[i]  = hit ? Int32(1) : Int32(0)
    ts[i]    = hit ? t : -1f0
    insts[i] = inst
end

# Deterministic directions from `O` towards the unit cube around the origin.
function _probe_dirs(n, O)
    s = Ref(987)
    nr() = (s[] = (1103515245 * s[] + 12345) % 2147483648; Float32(s[]) / 2147483648f0)
    [Vec3f(LinearAlgebra.normalize(Point3f(2*(nr()-0.5f0), 2*(nr()-0.5f0), 0) - O)) for _ in 1:n]
end

@testset "Metal: the hit-record ABI matches the MSL it is shared with" begin
    @test sizeof(MEXT.MetalHit) == 24
    @test isbitstype(MEXT.MetalHit)
    @test fieldnames(MEXT.MetalHit) == (:t, :hit, :prim, :inst, :bu, :bv)
    # Plain scalars, no padding — the MSL struct is declared the same way.
    @test all(fieldoffset(MEXT.MetalHit, i) == 4(i - 1) for i in 1:6)
end

@testset "Metal: HWTLAS is reachable through the portable constructor" begin
    be = Metal.MetalBackend()
    # `HWTLAS{Tri}(backend)` is what Hikari's `default_accel` calls; a caller
    # must never name the backend's own type.
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    @test t isa Mantle.HWTLAS{Raycore.Triangle{UInt32}}
    @test Raycore.n_geometries(t) == 0
    @test Raycore.n_instances(t) == 0

    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
    h = push!(t, mesh)
    @test h isa Raycore.TLASHandle
    @test Raycore.n_geometries(t) == 1

    Raycore.sync!(t)
    @test Raycore.n_instances(t) == 1
    b = Raycore.world_bound(t)
    @test all(b.p_min .< -0.9f0) && all(b.p_max .> 0.9f0)

    # `sync!` is the sole owner of the adapted form, and a second one with
    # nothing dirty must not rebuild.
    a1 = Adapt.adapt(be, t)
    a2 = Adapt.adapt(be, t)
    @test a1 === a2
end

@testset "Metal: traversal agrees with the software BVH" begin
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
    O = Point3f(0, 0, 4)
    n = 2048
    dirs = _probe_dirs(n, O)

    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, mesh); Raycore.sync!(hw)
    accel_hw = Adapt.adapt(be, hw)

    d_d = KA.allocate(be, Vec3f, n); copyto!(d_d, dirs)
    h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
    _hw_probe!(be)(h_d, t_d, i_d, d_d, accel_hw, O; ndrange = n)
    KA.synchronize(be)
    hh, ht, hi = Array(h_d), Array(t_d), Array(i_d)

    sw = Raycore.TLAS(KA.CPU())
    push!(sw, mesh); Raycore.sync!(sw)
    accel_sw = Adapt.adapt(KA.CPU(), sw)
    sh = zeros(Int32, n); st = zeros(Float32, n); si = zeros(UInt32, n)
    _hw_probe!(KA.CPU())(sh, st, si, copy(dirs), accel_sw, O; ndrange = n)
    KA.synchronize(KA.CPU())

    # Something must actually be hit, or every assertion below is vacuous.
    @test sum(sh) > n ÷ 4
    # Residency failure shows up exactly here: all-miss on the hardware side
    # while the software side hits.
    @test sum(hh) == sum(sh)
    @test hh == sh
    @test maximum(abs.(ht[hh .== 1] .- st[sh .== 1])) < 1e-3

    # The instance CUSTOM index, which no `push!` in this test sets, so 0 —
    # NOT Metal's instance_id.
    @test all(==(UInt32(0)), hi[hh .== 1])
end

@testset "Metal: several instances, and the custom index stays 0" begin
    # Two instances of one mesh, offset along x. A traversal that returned
    # `instance_id` here would give 1 for the second instance, and Hikari would
    # look up a medium interface that does not exist.
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0))
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(t, mesh, Mat4f(I))
    shifted = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, 1.5f0,0,0,1)   # column-major translate
    push!(t, mesh, shifted)
    Raycore.sync!(t)
    @test Raycore.n_geometries(t) == 2
    @test Raycore.n_instances(t) == 2
    accel = Adapt.adapt(be, t)

    # One ray at each sphere.
    O = Point3f(0, 0, 4)
    dirs = [Vec3f(0, 0, -1), Vec3f(LinearAlgebra.normalize(Point3f(1.5f0, 0, 0) - O))]
    n = length(dirs)
    d_d = KA.allocate(be, Vec3f, n); copyto!(d_d, dirs)
    h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
    _hw_probe!(be)(h_d, t_d, i_d, d_d, accel, O; ndrange = n)
    KA.synchronize(be)
    @test Array(h_d) == Int32[1, 1]              # both instances are hit
    @test all(==(UInt32(0)), Array(i_d))         # …and neither reports a custom index
end

@testset "Metal: an empty structure syncs and misses" begin
    be = Metal.MetalBackend()
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    Raycore.sync!(t)
    @test Raycore.n_instances(t) == 0
    @test Adapt.adapt(be, t) isa Mantle.AdaptedAccel
end

@testset "Metal: no ray-tracing pipeline, and Hikari must be told so" begin
    # Metal traverses from inside the shading kernel, so there is no shader
    # binding table. Hikari picks its trace path off exactly this.
    be = Metal.MetalBackend()
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    @test Mantle.supports_rt_pipeline(be) == false
    @test Mantle.supports_rt_pipeline(t) == false
    push!(t, GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0)))
    Raycore.sync!(t)
    @test Mantle.supports_rt_pipeline(Adapt.adapt(be, t)) == false
end
