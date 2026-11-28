# `Raycore.closest_hit` means the same on Raycore's software structure
# (`Raycore.TLAS`) and on the hardware one (`Mantle.HWTLAS`): one scene, pushed
# into both with the same transforms and custom indices and traced by the same
# kernel, gives the same hits, the same custom indices, the same distances and
# the same hit triangles in world space. And again after one kernel rewrote both
# structures' instance records (`Raycore.instance_buffer`) and `Raycore.refit!`
# committed them.
#
# The two used to disagree on two of the five values: the software structure
# returned the instance's position in its instance array and the triangle as its
# BLAS stores it. A kernel written against one read the wrong id, or the wrong
# space, on the other.

using Test, Mantle, Raycore, Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Point3f, Vec3f, Mat4f, GLTriangleFace, Rect3f
import GeometryBasics
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

# Record `i` gets `transforms[i]`, `ids[i]` and `mask`: what a simulation writes.
@kernel function samehit_writerecords!(records, @Const(transforms), @Const(ids), mask)
    i = @index(Global)
    @inbounds records[i] = Raycore.InstanceRecord(transforms[i], ids[i], mask)
end

"""Column-major affine `Mat4f`: scale `s`, then a turn by `θ` about z, then a move to `p`."""
function placed(p; s = Vec3f(1, 1, 1), θ = 0f0)
    c, n = cos(θ), sin(θ)
    Mat4f(c * s[1], n * s[1], 0, 0, -n * s[2], c * s[2], 0, 0, 0, 0, s[3], 0, p[1], p[2], p[3], 1)
end

"""The triangles of `mesh` moved by `m`, as vertex triples, in the order a BLAS holds them."""
function worldtriangles(mesh, m)
    em = GeometryBasics.expand_faceviews(mesh)
    vs = GeometryBasics.decompose(Point3f, em)
    move(v) = Point3f((m * GeometryBasics.Vec4f(v[1], v[2], v[3], 1))[1:3]...)
    return [map(k -> move(vs[k]), (f[1], f[2], f[3])) for f in GeometryBasics.decompose(GLTriangleFace, em)]
end

"""The triangles of `tris` with every vertex at the top: what a ray from above meets first."""
toptriangles(tris) = (z = maximum(t -> maximum(v -> v[3], t), tris);
                      filter(t -> all(v -> isapprox(v[3], z; atol = 1f-5), t), tris))

"""
Rays at the centroid of every top triangle of every instance: straight down, and
slanted. `want` is, per ray, the custom index and world centroid it should hit,
or `nothing` for the two rays aimed past everything.
"""
function aimat(instances)
    origins, dirs, want = Point3f[], Vec3f[], Any[]
    for (tris, id) in instances, t in toptriangles(tris)
        c = tricentroid(t)
        push!(origins, c + Vec3f(0, 0, 5)); push!(dirs, Vec3f(0, 0, -1)); push!(want, (id, c))
        o = c + Vec3f(0.15f0, 0.1f0, 3)
        push!(origins, o); push!(dirs, Vec3f(c - o)); push!(want, nothing)
    end
    for o in (Point3f(40, 40, 5), Point3f(-40, 0, 5))
        push!(origins, o); push!(dirs, Vec3f(0, 0, -1)); push!(want, nothing)
    end
    return origins, dirs, want
end

"""Software and hardware traced `probe` alike: hits, ids, distances, centroids."""
function samehits(probe, sw, hw; mask = nothing)
    s = tlastrace(probe, sw; mask)
    h = tlastrace(probe, hw; mask)
    both = s.hit .& h.hit
    @test s.hit == h.hit
    @test s.id == h.id
    @test all(isapprox.(s.t[both], h.t[both]; atol = 1f-3))
    @test all(i -> isapprox(s.centroid[i], h.centroid[i]; atol = 1f-3), findall(both))
    return s
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "Raycore.TLAS and HWTLAS return the same hit" begin
    tri = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
        [Point3f(-0.4, -0.4, 0), Point3f(0.4, -0.4, 0), Point3f(0, 0.4, 0)], [GLTriangleFace(1, 2, 3)]))
    box = GeometryBasics.normal_mesh(Rect3f(Vec3f(-0.3), Vec3f(0.6)))
    # Three instances of one triangle: moved, stretched in x, turned about z.
    tris_at = [placed(Point3f(0, 0, 0)), placed(Point3f(2, 0, 0.2); s = Vec3f(1.5, 1, 1)),
               placed(Point3f(4, 0, 0.5); θ = 0.5f0)]
    # A box squashed and stretched, with an id past 24 bits: both structures
    # report its low 24, all a hardware record holds.
    box_at = placed(Point3f(6, 0, 0); s = Vec3f(1, 2, 0.5))
    build(tlas) = (push!(tlas, tri, tris_at; instance_ids = UInt32[11, 12, 13]),
                   push!(tlas, box, box_at; instance_id = UInt32(0x01000007)))
    sw = Raycore.TLAS(TESTBACKEND)
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    sw_tris, sw_box = build(sw)
    hw_tris, hw_box = build(hw)
    Raycore.sync!(sw)
    Raycore.sync!(hw)

    scene = [[(worldtriangles(tri, m), UInt32(id)) for (m, id) in zip(tris_at, 11:13)];
             (worldtriangles(box, box_at), UInt32(7))]
    origins, dirs, want = aimat(scene)
    probe = TLASProbe(TESTBACKEND, origins, dirs)
    r = samehits(probe, sw, hw)
    down = findall(!isnothing, want)
    # Not a vacuous agreement: every downward ray met the triangle it was aimed
    # at, the instance's custom index came with it, and the two aimed past
    # everything missed.
    @test all(r.hit[down])
    @test r.id[down] == [first(want[i]) for i in down]
    @test all(i -> isapprox(r.centroid[i], last(want[i]); atol = 1f-4), down)
    @test all(i -> isapprox(r.t[i], 5f0; atol = 1f-4), down)
    @test !any(r.hit[end-1:end]) && all(iszero, r.id[end-1:end])

    # One kernel rewrites the three triangles' records in both structures: moved
    # 3 along y, new ids, mask 0x02. Neither takes them before `refit!`.
    moved = [placed(Point3f(0, 3, 0)), placed(Point3f(2, 3, 0.2); θ = -0.3f0),
             placed(Point3f(4, 3, 0.5); s = Vec3f(0.5, 2, 1))]
    for (tlas, h) in ((sw, sw_tris), (hw, hw_tris))
        samehit_writerecords!(TESTBACKEND)(Raycore.instance_buffer(tlas, h),
            Mantle.devicearray(TESTBACKEND, Raycore.mat4_to_mat3x4.(moved)),
            Mantle.devicearray(TESTBACKEND, UInt32[21, 22, 23]), UInt32(0x02); ndrange = 3)
    end
    KA.synchronize(TESTBACKEND)
    @test samehits(probe, sw, hw).id == r.id
    @test Raycore.refit!(sw) === sw
    @test Raycore.refit!(hw) === hw

    scene = [[(worldtriangles(tri, m), UInt32(id)) for (m, id) in zip(moved, 21:23)];
             (worldtriangles(box, box_at), UInt32(7))]
    origins, dirs, want = aimat(scene)
    probe = TLASProbe(TESTBACKEND, origins, dirs)
    r = samehits(probe, sw, hw)
    down = findall(!isnothing, want)
    @test all(r.hit[down])
    @test r.id[down] == [first(want[i]) for i in down]
    @test all(i -> isapprox(r.centroid[i], last(want[i]); atol = 1f-4), down)
    # The mask the kernel wrote: a ray with mask 0x01 sees only the box.
    m = samehits(probe, sw, hw; mask = 0x01)
    @test m.id[down] == [first(want[i]) == 7 ? UInt32(7) : UInt32(0) for i in down]
    # Nothing is left where the triangles were.
    @test !any(samehits(TLASProbe(TESTBACKEND, [Point3f(x, 0, 5) for x in (0, 2, 4)],
                                  fill(Vec3f(0, 0, -1), 3)), sw, hw).hit)
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end
