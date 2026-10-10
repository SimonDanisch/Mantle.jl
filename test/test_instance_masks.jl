# Per-instance cull masks, chosen per RAY: `closest_hit(accel, ray, mask)` and
# `any_hit(accel, ray, mask)` see an instance only when its mask (`instance_mask`
# at `push!`) shares a bit with the ray's.
#
# Two single-triangle instances stacked along the ray, z = 5 with mask 0x01 and
# z = 10 with mask 0x02, and one ray per mask: 0x01 must hit the near one, 0x02
# pass through it to the far one, 0xFF hit the near one, 0x04 nothing. Run against
# the hardware structure and against Raycore's own software TLAS, in the same
# kernel: the mask has to mean the same thing on both. Both hardware backends
# took a mask and hardcoded 0xFF until 2026-10-10, so a mask-0x02 ray hit the
# near instance on every one of them.

using Test, Mantle, Raycore, Adapt
using KernelAbstractions
using GeometryBasics: Point3f, Vec3f, Mat4f, GLTriangleFace
import GeometryBasics
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

const Tri = Raycore.Triangle{UInt32}

function tri_mesh(z::Float32)
    verts = [Point3f(-1f0, -1f0, z), Point3f(1f0, -1f0, z), Point3f(0f0, 1f0, z)]
    GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, [GLTriangleFace(1, 2, 3)]))
end

# Ray `i` along +z from the origin with cull mask `masks[i]`; the two-argument
# forms beside it, which are mask 0xFF.
@kernel function mask_probe!(closest, anyhit, plain, plainany, @Const(masks), accel)
    i = @index(Global)
    ray = Raycore.Ray(o = Point3f(0f0, 0f0, 0f0), d = Vec3f(0f0, 0f0, 1f0), t_max = 1f3)
    hit, _, t, _, _ = Raycore.closest_hit(accel, ray, masks[i])
    closest[i] = hit ? t : -1f0
    hit, _, t, _, _ = Raycore.any_hit(accel, ray, masks[i])
    anyhit[i] = hit ? t : -1f0
    hit, _, t, _, _ = Raycore.closest_hit(accel, ray)
    plain[i] = hit ? t : -1f0
    hit, _, t, _, _ = Raycore.any_hit(accel, ray)
    plainany[i] = hit ? t : -1f0
end

const MASKS = UInt32[0x01, 0x02, 0xFF, 0x04, 0x100]

function trace_masks(accel)
    n = length(MASKS)
    out = [Mantle.devicearray(TESTBACKEND, fill(-2f0, n)) for _ in 1:4]
    mask_probe!(TESTBACKEND)(out..., Mantle.devicearray(TESTBACKEND, MASKS),
                             Adapt.adapt(TESTBACKEND, accel); ndrange = n)
    KA.synchronize(TESTBACKEND)
    return map(Array, out)
end

function check_masks(accel)
    closest, anyhit, plain, plainany = trace_masks(accel)
    @test isapprox(closest[1], 5f0; atol = 1f-3)    # 0x01 sees only the near one
    @test isapprox(closest[2], 10f0; atol = 1f-3)   # 0x02 passes through it
    @test isapprox(closest[3], 5f0; atol = 1f-3)    # 0xFF sees both, the near one wins
    @test closest[4] == -1f0                        # 0x04 sees neither
    @test closest[5] == -1f0                        # only the low 8 bits count
    # `any_hit` takes the same mask; with one instance in view the hit is that one.
    @test isapprox(anyhit[1], 5f0; atol = 1f-3)
    @test isapprox(anyhit[2], 10f0; atol = 1f-3)
    @test anyhit[3] > 0f0
    @test anyhit[4] == -1f0
    @test anyhit[5] == -1f0
    # The two-argument forms are mask 0xFF.
    @test all(t -> isapprox(t, 5f0; atol = 1f-3), plain)
    @test all(>(0f0), plainany)
end

@testset "the software TLAS: a ray sees an instance only through a common mask bit" begin
    sw = Raycore.TLAS(TESTBACKEND)
    push!(sw, tri_mesh(5f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(1), instance_mask = UInt8(0x01))
    push!(sw, tri_mesh(10f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(2), instance_mask = UInt8(0x02))
    Raycore.sync!(sw)
    check_masks(sw)
end

if Mantle.supports_hwtlas(TESTBACKEND)
    @testset "the hardware TLAS: a ray sees an instance only through a common mask bit" begin
        hw = Mantle.HWTLAS{Tri}(TESTBACKEND)
        push!(hw, tri_mesh(5f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(1), instance_mask = UInt8(0x01))
        push!(hw, tri_mesh(10f0), Mat4f(LinearAlgebra.I); instance_id = UInt32(2), instance_mask = UInt8(0x02))
        Raycore.sync!(hw)
        check_masks(hw)
    end
else
    @info "no hardware acceleration structures on this backend; the hardware half is skipped"
end
