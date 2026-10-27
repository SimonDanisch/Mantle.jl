"""
Procedural geometry TRACED: boxes in a bottom-level structure, answered by a
Julia payload through `Mantle.procedural_candidate`, read back from
`Raycore.closest_hit` like any triangle hit. Gated on `supports_hwtlas` and
`supports_procedural_traversal`.

The payload solves a sphere inside each box, with a radius per box read from a
device array, so the hit has a real distance and a cell, and the rays meet
several boxes so the NEAREST sphere has to win. The payload's best hit makes a
four-float round trip through the hit record, where a dropped or reordered slot
still traces and reports a miss or an off-by-one cell. Hence both the distance and
the cell are asserted.

Point location among overlapping boxes, and a box-only structure that still
builds, syncs and adapts, are `test_aabb_blas_overlap.jl` and
`test_hwtlas_batch_triangles.jl`.
"""

using Test, Mantle, Raycore, GeometryBasics, KernelAbstractions
using Adapt, StaticArrays
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""A payload: one radius per box, in a device array, and whether its spheres are opaque."""
struct RadiusElems{A}
    r::A
    opaque::Bool
end
Adapt.adapt_structure(to, e::RadiusElems) = RadiusElems(Adapt.adapt(to, e.r), e.opaque)

struct RadiusHit
    t::Float32
    ξ₁::Float32
    ξ₂::Float32
    cell::UInt32
end

Mantle.procedural_miss(::RadiusElems) = RadiusHit(1f30, 0f0, 0f0, UInt32(0))
Mantle.procedural_opaque(e::RadiusElems) = e.opaque

# Written exactly as `src/raytracing/accel.jl` documents: solve in OBJECT space,
# ask the QUERY for the ray, and answer only a `t` that improves on `best`; the
# traversal commits it. Box `cell` (1-based) is centred on `(2 * (cell - 1), 0, 0)`.
function Mantle.procedural_candidate(e::RadiusElems, best::RadiusHit, q)
    cell = Mantle.candidate_primitive_index(q)
    o, d = Mantle.candidate_object_ray(q)
    @inbounds rad = e.r[cell]
    ex = o[1] - 2f0 * Float32(cell - 1); ey = o[2]; ez = o[3]
    b = ex*d[1] + ey*d[2] + ez*d[3]
    c = ex*ex + ey*ey + ez*ez - rad*rad
    disc = b*b - c
    if disc >= 0f0
        t = -b - sqrt(disc)
        if t > 1f-4 && t < best.t
            return RadiusHit(t, 0f0, 0f0, cell % UInt32)
        end
    end
    return best
end

Mantle.procedural_commit(::RadiusElems, ::RadiusHit, prim, t, empty) = empty
# The cell rides out in the barycentric's first slot, which is where a payload's
# own parameter travels — see `procedural_bary`.
Mantle.procedural_bary(::RadiusElems, b::RadiusHit) =
    SVector{3,Float32}(Float32(b.cell), 0f0, 0f0)

@kernel function closest_hit_probe!(hits, ts, cells, @Const(ys), accel, ox)
    i = @index(Global)
    r = Raycore.Ray(o = Point3f(ox, ys[i], 0f0), d = Vec3f(1, 0, 0), t_max = 1f30)
    hit, prim, t, bary, inst = Raycore.closest_hit(accel, r)
    @inbounds hits[i]  = hit ? Int32(1) : Int32(0)
    @inbounds ts[i]    = hit ? t : -1f0
    @inbounds cells[i] = hit ? UInt32(bary[1]) : UInt32(0)
end

# `Raycore.transparent_hits` of the same rays, keeping four.
@kernel function transparent_probe!(out, @Const(ys), accel, ox, limit)
    i = @index(Global)
    r = Raycore.Ray(o = Point3f(ox, ys[i], 0f0), d = Vec3f(1, 0, 0), t_max = 1f30)
    @inbounds out[i] = Raycore.transparent_hits(accel, r, 0xff, Val(4), limit)
end

@testset "procedural geometry" begin
    be = TESTBACKEND

    @testset "the capability is answered on every handle" begin
        # Building boxes and tracing them are separate questions, and this one
        # answers the second. The default is `false`, so a backend opts in and
        # nothing traces boxes by accident.
        @test Mantle.supports_procedural_traversal(nothing) == false
        if Mantle.supports_hwtlas(be)
            t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
            @test Mantle.supports_procedural_traversal(t) == Mantle.supports_procedural_traversal(be)
            push!(t, GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0)))
            Raycore.sync!(t)
            # …and an `AdaptedAccel` answers too, rather than falling to that
            # default. A caller that has to ask usually holds one of these;
            # falling through would give a WRONG answer on a backend that does
            # trace boxes.
            @test Mantle.supports_procedural_traversal(Adapt.adapt(be, t)) ==
                  Mantle.supports_procedural_traversal(be)
        end
    end

    @testset "closest_hit traces procedural geometry" begin
        if !(Mantle.supports_hwtlas(be) && Mantle.supports_procedural_traversal(be))
            @info "no procedural hardware traversal on this backend; skipping"
            @test_skip Mantle.supports_procedural_traversal(be)
        else
            radii = Float32[0.75, 0.5, 0.9, 0.6]
            nb = length(radii)
            aabbs = [Mantle.AABB(Point3f(2f0*i - 1, -1, -1), Point3f(2f0*i + 1, 1, 1))
                     for i in 0:(nb - 1)]
            blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(be))) do ctx
                Mantle.build_blas_aabb(ctx, aabbs)
            end
            hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
            push!(hw, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(0))
            # The payload is the field Hikari's `fem.jl` sets; `sync!` builds what
            # the traversal needs from it.
            hw.procedural = RadiusElems(Mantle.devicearray(be, radii), true)
            Raycore.sync!(hw)

            ys = Float32[0.0, 0.4, 0.8, 1.2]
            n  = length(ys)
            dy = Mantle.devicearray(be, ys)
            dh = KA.zeros(be, Int32, n); dt = KA.zeros(be, Float32, n); dc = KA.zeros(be, UInt32, n)
            closest_hit_probe!(be, 64)(dh, dt, dc, dy, Adapt.adapt(be, hw), -5f0; ndrange = n)
            KA.synchronize(be)
            gh, gt, gc = Array(dh), Array(dt), Array(dc)

            # The nearest sphere along +x from -5, per ray, on the host. PER-BOX
            # radii, so this also checks the payload's device array really was
            # read: a traversal that ignored `e.r` would agree only for box 0.
            for (i, y) in enumerate(ys)
                bt = Inf32; bc = 0
                for p in 0:(nb - 1)
                    ex = -5f0 - 2f0*p
                    disc = ex*ex - (ex*ex + y*y - radii[p + 1]^2)
                    disc < 0f0 && continue
                    t = -ex - sqrt(disc)
                    (t > 1f-4 && t < bt) && (bt = t; bc = p + 1)
                end
                if bc == 0
                    @test gh[i] == Int32(0)
                else
                    @test gh[i] == Int32(1)
                    @test gt[i] ≈ bt rtol=1f-5
                    # 1-BASED, from the payload's own `cell`. Rebuilding this from
                    # the primitive id instead is off by one, and `t` stays right —
                    # so a test that checked only the distance would pass.
                    @test Int(gc[i]) == bc
                end
            end
        end
    end
    @testset "transparent_hits records a see-through payload's hits, stops at an opaque one's" begin
        if !(Mantle.supports_hwtlas(be) && Mantle.supports_procedural_traversal(be))
            @test_skip Mantle.supports_procedural_traversal(be)
        else
            # Five spheres along +x; a ray down the axis meets all five, at their
            # near sides. A shadow ray through water (a level set with `shadows`)
            # is this: each surface it crosses is recorded and shaded by the caller.
            radii = Float32[0.75, 0.5, 0.9, 0.6, 0.8]
            nb = length(radii)
            aabbs = [Mantle.AABB(Point3f(2f0*i - 1, -1, -1), Point3f(2f0*i + 1, 1, 1)) for i in 0:(nb - 1)]
            blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(be))) do ctx
                Mantle.build_blas_aabb(ctx, aabbs)
            end
            dy = Mantle.devicearray(be, Float32[0])
            # One structure per payload: the payload is set with the push that
            # makes the structure rebuild, as Hikari does.
            function shoot(opaque; limit = 10)
                hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
                push!(hw, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(7))
                hw.procedural = RadiusElems(Mantle.devicearray(be, radii), opaque)
                Raycore.sync!(hw)
                out = Adapt.adapt(be, [Raycore.TransparentHits{4}()])
                transparent_probe!(be, 64)(out, dy, Adapt.adapt(be, hw), -5f0, limit; ndrange = 1)
                KA.synchronize(be)
                return only(Array(out))
            end
            @test shoot(true).blocked
            th = shoot(false)
            near = [5f0 + 2f0 * (p - 1) - radii[p] for p in 1:nb]
            @test !th.blocked && th.count == nb
            @test [h.t for h in th.hits] ≈ near[1:4] rtol = 1f-5
            @test all(h -> h.kind == Raycore.HIT_GENERATED && h.custom == 7, th.hits)
            # The payload's best rides in the record: the cell, 1-based.
            @test [Int(Mantle.unpackbest(RadiusHit, h.a).cell) for h in th.hits] == 1:4
            @test shoot(false; limit = 4).blocked
        end
    end
end
