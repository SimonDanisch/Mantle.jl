# Gate on `slide.jl`: every function on the slide must agree with the one the
# demo actually runs. A slide that has drifted from the code is worse than no
# slide, and this is two seconds of CPU.

using Test, GeometryBasics, LinearAlgebra, Random
import Hikari

# The three things the slide's signatures name and the talk does not show:
# the mesh shader's output handle, a ray, and what an intersection returns.
struct MeshOut; verts::Vector{Any}; tris::Vector{Any}; end
MeshOut() = MeshOut(Any[], Any[])
struct Ray; o::Vec3f; d::Vec3f; end
struct Hit; t::Float32; ξ::Vec2f; n::Vec3f; end
module KI
    set_mesh_vertex!(out, i, v) = (push!(out.verts, (i, v)); nothing)
    set_mesh_triangle!(out, i, ijk) = (push!(out.tris, (i, ijk)); nothing)
end

include("slide.jl")

# Curved elements: a plane with a quadratic bump, so a single Newton seed
# converges. (The demo's tracer uses nine — see `Hikari.NEWTON_SEEDS`.)
Random.seed!(20260924)
const NCELLS = 4
cx = zeros(Float32, 9, NCELLS); cy = zeros(Float32, 9, NCELLS); cu = zeros(Float32, 9, NCELLS)
for c in 1:NCELLS
    cx[:, c] .= Float32.(0.05 .* randn(9));  cx[2, c] += 1f0   # x ≈ ξ₁
    cy[:, c] .= Float32.(0.05 .* randn(9));  cy[4, c] += 1f0   # y ≈ ξ₂
    cu[:, c] .= Float32.(0.20 .* randn(9))                     # the solution
end
# The slide's `Element` is ONE cell; Hikari indexes a block by cell.
cell(c) = Element(view(cx, :, c), view(cy, :, c), view(cu, :, c))
const XIS = [Vec2f(a, b) for a in -1:0.25f0:1 for b in -1:0.25f0:1]

@testset "slide.jl == the code the demo runs" begin
    @testset "the element: x(ξ), ∂x/∂ξ and the normal" begin
        for c in 1:NCELLS, ξ in XIS
            e = cell(c)
            a, b = Float64(ξ[1]), Float64(ξ[2])
            # `warp = 1`: the slide folds the demo's display scale into u(ξ).
            p = Hikari.surfacepoint(cx, cy, cu, c, a, b, 1.0)
            ta, tb = Hikari.surfacetangents(cx, cy, cu, c, a, b, 1.0)
            n = Hikari.surfacenormal(cx, cy, cu, c, a, b, 1.0)
            @test surfacepoint(e, ξ) ≈ Vec3f(p...)      rtol=1f-5
            @test jacobian(e, ξ)[:, 1] ≈ Vec3f(ta...)   rtol=1f-5
            @test jacobian(e, ξ)[:, 2] ≈ Vec3f(tb...)   rtol=1f-5
            @test normal(e, ξ) ≈ n                      rtol=1f-4
        end
    end

    # The demo's own key decode, vendored here so the test needs no GPU.
    # `mantle_isubd.jl:133`, unedited apart from the `basecorners` lookup.
    refkey_base(k) = Int(k >> 32)
    function refkey_corners(k::UInt64, c1, c2, c3)
        w1 = (1.0, 0.0, 0.0); w2 = (0.0, 1.0, 0.0); w3 = (0.0, 0.0, 1.0)
        leb = k % UInt32
        for i in (31 - leading_zeros(leb) - 1):-1:0
            m = ((w1[1] + w3[1]) / 2, (w1[2] + w3[2]) / 2, (w1[3] + w3[3]) / 2)
            if ((leb >> i) & 1) == UInt32(0)
                w1, w2, w3 = w1, m, w2
            else
                w1, w2, w3 = w2, m, w3
            end
        end
        mix(w) = (w[1] * c1[1] + w[2] * c2[1] + w[3] * c3[1],
                  w[1] * c1[2] + w[2] * c2[2] + w[3] * c3[2])
        return mix(w1), mix(w2), mix(w3)
    end

    @testset "a key decodes to the demo's triangle" begin
        C = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0))
        root = (UInt64(1) << 32) | UInt64(1)          # base triangle 1, depth 0
        keys = [root];  frontier = [root]             # every path to depth 6
        for _ in 1:6
            frontier = [child(k, i) for k in frontier for i in 0:1]
            append!(keys, frontier)
        end
        @test length(keys) == 2^7 - 1 == length(unique(keys))
        for k in keys
            @test depth(k) == 31 - leading_zeros(k % UInt32)
            @test refkey_base(k) == 1                 # `child` leaves it alone
            got = corners(k, C...)
            want = refkey_corners(k, C...)
            for j in 1:3
                @test got[j] ≈ Vec2f(want[j]...)      rtol=1f-6
            end
        end
        # Bisection halves the area, so depth d triangles are 2^-d of the base.
        area(t) = abs((t[2][1]-t[1][1])*(t[3][2]-t[1][2]) -
                      (t[3][1]-t[1][1])*(t[2][2]-t[1][2])) / 2
        a0 = area(corners(root, C...))
        for k in keys
            @test area(corners(k, C...)) ≈ a0 / 2.0^depth(k)   rtol=1f-5
        end
    end

    # The demo's own three-way decision, ported from `classify_kernel!` and
    # `scatter_kernel!` (mantle_isubd.jl:262, :279) with `max_depth` dropped and
    # `excess_levels(k) > 0` read as `chorderror(k) > tol` — so the ESTIMATOR is
    # held identical and only the branch structure is under test.
    function refcounts(e, k::Key, tol)
        coarse(x) = chorderror(e, corners(x)) > tol
        d = depth(k)
        if coarse(k)
            return (child(k, 0), child(k, 1))
        elseif d > 0 && !coarse(parent(k))
            return firstborn(k) ? (parent(k),) : ()
        else
            return (k,)
        end
    end

    @testset "the compute overload splits, merges and keeps like the demo" begin
        root = (UInt64(1) << 32) | UInt64(1)
        keys = [root];  frontier = [root]
        for _ in 1:5
            frontier = [child(k, i) for k in frontier for i in 0:1]
            append!(keys, frontier)
        end
        seen = Set{Symbol}()
        for c in 1:NCELLS, tol in Float32[1f-4, 1f-3, 1f-2, 1f-1, 1f0], k in keys
            e = cell(c)
            out = Keys()
            resolve(out, e, k, tol)
            @test Tuple(out.v) == refcounts(e, k, tol)
            n = length(out.v)
            @test n in (0, 1, 2)
            push!(seen, n == 2 ? :split : (n == 0 ? :drop :
                  (only(out.v) == k ? :keep : :merge)))
        end
        # All four outcomes must actually occur, or the test proves nothing.
        @test seen == Set([:split, :merge, :keep, :drop])
    end

    @testset "refining to a tolerance converges and honours it" begin
        e = cell(1)
        for tol in Float32[3f-1, 1f-1, 3f-2]
            keys = [(UInt64(1) << 32) | UInt64(1)]
            for _ in 1:14                         # to a fixed point, like `refine`
                out = Keys()
                for k in keys
                    resolve(out, e, k, tol)
                end
                out.v == keys && break
                keys = out.v
            end
            @test !isempty(keys)
            @test all(k -> chorderror(e, corners(k)) <= tol, keys)   # tolerance met
            # The keys tile the base triangle: areas sum to the root's.
            area(t) = abs((t[2][1]-t[1][1])*(t[3][2]-t[1][2]) -
                          (t[3][1]-t[1][1])*(t[2][2]-t[1][2])) / 2
            @test sum(k -> area(corners(k)), keys) ≈ 2.0    rtol=1f-5
        end
    end

    @testset "the mesh-shader overload emits the surface" begin
        out = MeshOut()
        e = cell(2)
        key = child(child((UInt64(1) << 32) | UInt64(1), 1), 0)
        ξ3 = corners(key)
        resolve(out, e, key, Mat4f(I))
        @test length(out.verts) == 3 && length(out.tris) == 1
        for (i, v) in out.verts
            @test v.position ≈ toclip(surfacepoint(e, ξ3[i]), Mat4f(I))
            @test v.normal   ≈ normal(e, ξ3[i])
            @test v.ξ == ξ3[i]
        end
    end

    @testset "the raytracing overload lands on the same surface" begin
        for c in 1:NCELLS, ξstar in [Vec2f(-0.7, 0.3), Vec2f(0, 0), Vec2f(0.5, -0.55)]
            e = cell(c)
            p = surfacepoint(e, ξstar)
            o = Vec3f(0.3, -0.2, 6)                 # a camera above the plate
            d = normalize(p - o)
            h = resolve(Ray(o, d), e)
            @test h.ξ ≈ ξstar                       atol=1f-4   # found the root
            @test o + h.t * d ≈ p                   atol=1f-4   # at the right t
            @test h.n ≈ normal(e, ξstar)            atol=1f-4   # analytic normal

            # And the demo's own solver, on the same ray, agrees.
            hit, t, ξa, ξb = Hikari.intersect_element(cx, cy, cu, c,
                Float64(o[1]), Float64(o[2]), Float64(o[3]),
                Float64(d[1]), Float64(d[2]), Float64(d[3]), 6.0, 1.0)
            @test hit
            @test Vec2f(ξa, ξb) ≈ ξstar             atol=1f-4
            @test Float32(t) ≈ h.t                  rtol=1f-4
        end
    end
end
