# The code for the talk: one element, three consumers, nothing written twice.
#
# Everything between the two rules is THE SLIDE, and carries no comments by
# design. Below it are the helpers it names, and `test_slide.jl` checks all of
# it against the functions the demo runs.

using GeometryBasics, LinearAlgebra

"""One curved element: nine monomial coefficients for each of x, y and u."""
struct Element{C}
    x::C
    y::C
    u::C
end

"""One triangle of a subdivided element: its base triangle and the path of
longest-edge bisections below it, in a single `UInt64`."""
const Key = UInt64

"""The key array the adaptive pass writes."""
struct Keys
    v::Vector{Key}
end
Keys() = Keys(Key[])

# ═══════════════════════════════════════════════════════════════════════════

function resolve(out::Keys, e::Element, key::Key, tol::Float32)
    toocoarse(k) = chorderror(e, corners(k)) > tol
    if toocoarse(key)
        emit!(out, children(key)...)
    elseif hasparent(key) && !toocoarse(parent(key))
        firstborn(key) && emit!(out, parent(key))
    else
        emit!(out, key)
    end
end

function resolve(out::MeshOut, e::Element, key::Key, mvp::Mat4f)
    ξ = corners(key)
    for i in 1:3
        KI.set_mesh_vertex!(out, i, (position = toclip(surfacepoint(e, ξ[i]), mvp),
                                     normal   = normal(e, ξ[i]),
                                     ξ = ξ[i]))
    end
    KI.set_mesh_triangle!(out, 1, (1, 2, 3))
end

function resolve(ray::Ray, e::Element; steps = 8)
    ξ, t = Vec2f(0, 0), 0f0
    for _ in 1:steps
        residual = surfacepoint(e, ξ) - ray(t)
        Δξ₁, Δξ₂, Δt = hcat(jacobian(e, ξ), -ray.d) \ residual
        ξ -= Vec2f(Δξ₁, Δξ₂)
        t -= Δt
    end
    return Hit(t, ξ, normal(e, ξ))
end

# ═══════════════════════════════════════════════════════════════════════════
#
# Not on the slide. First, what `Element` can do: a quadratic quad in the
# monomial basis the reference solver hands over — nine coefficients each.

mono( a, b) = (1, a, a^2,  b, a*b, a^2*b,  b^2, a*b^2, a^2*b^2)
monoa(a, b) = (0, 1, 2a,   0, b,   2a*b,   0,   b^2,   2a*b^2)   # ∂/∂ξ₁
monob(a, b) = (0, 0, 0,    1, a,   a^2,    2b,  2a*b,  2a^2*b)   # ∂/∂ξ₂

poly(c, m) = sum(ntuple(i -> c[i] * m[i], Val(9)))

function surfacepoint(e::Element, ξ)
    m = mono(ξ[1], ξ[2])
    return Vec3f(poly(e.x, m), poly(e.y, m), poly(e.u, m))
end

function jacobian(e::Element, ξ)
    a = monoa(ξ[1], ξ[2]);  b = monob(ξ[1], ξ[2])
    return hcat(Vec3f(poly(e.x, a), poly(e.y, a), poly(e.u, a)),
                Vec3f(poly(e.x, b), poly(e.y, b), poly(e.u, b)))
end

# `LinearAlgebra.normalize`, QUALIFIED. This file is `include`d into `Main`
# alongside others, and once some earlier one has touched an undefined
# `Main.normalize` the binding is resolved and a later `using LinearAlgebra`
# cannot introduce it — an `UndefVarError` whose cause is the ORDER of the
# includes, not this file. Same reason `test/metal/test_hwtlas_metal.jl`
# qualifies it.
normal(e::Element, ξ) = (J = jacobian(e, ξ); LinearAlgebra.normalize(J[:, 1] × J[:, 2]))

toclip(x, mvp::Mat4f) = mvp * Vec4f(x[1], x[2], x[3], 1)

(ray::Ray)(t) = ray.o + t * ray.d

# The three edge midpoints and the centroid, in barycentric coordinates — where
# the chord is compared against the surface. The demo measures the GEOMETRY and
# the FIELD separately, against their own tolerances; one 3D distance covers
# both here because `surfacepoint` already carries u(ξ) out of plane.
# `f0`-suffixed, not bare. A bare `0.5` is a `Float64` literal, and these reach a
# GPU kernel through `chorderror` — on a device with no double precision that is
# "unsupported use of double value" at compile time, not a slow path. The values
# are exact in single either way; `1/3` is the only one that rounds, and it is a
# sample position, not a tolerance.
const SAMPLES = ((0.5f0, 0f0, 0.5f0), (0.5f0, 0.5f0, 0f0), (0f0, 0.5f0, 0.5f0),
                 (1f0/3f0, 1f0/3f0, 1f0/3f0))

function chorderror(e::Element, ξ::NTuple{3, Vec2f})
    x = map(ξi -> surfacepoint(e, ξi), ξ)
    return maximum(ntuple(Val(length(SAMPLES))) do s
        w = SAMPLES[s]
        chord = w[1] * x[1]  + w[2] * x[2]  + w[3] * x[3]     # the flat triangle
        exact = w[1] .* ξ[1] .+ w[2] .* ξ[2] .+ w[3] .* ξ[3]  # the same ξ, curved
        return norm(surfacepoint(e, exact) - chord)
    end)
end

# And the key. Parent and children are shifts, so nothing is stored per
# triangle and a key reconstructs its own geometry — which is what lets the
# mesh stage run with no vertex buffer and no index buffer.

const LEB = UInt64(typemax(UInt32))                     # the path lives here
withleb(k::Key, leb::UInt32) = (k & ~LEB) | UInt64(leb)

depth(k::Key)     = 31 - leading_zeros(k % UInt32)
hasparent(k::Key) = depth(k) > 0
parent(k::Key)    = withleb(k, (k % UInt32) >> 1)
child(k::Key, i)  = withleb(k, (k % UInt32) << 1 | UInt32(i))
children(k::Key)  = (child(k, 0), child(k, 1))
firstborn(k::Key) = iseven(k % UInt32)

"""The ξ a key stands for: one barycentric bisection folded per path bit."""
function corners(k::Key, c1 = (-1f0, -1f0), c2 = (1f0, -1f0), c3 = (1f0, 1f0))
    w1 = (1f0, 0f0, 0f0); w2 = (0f0, 1f0, 0f0); w3 = (0f0, 0f0, 1f0)
    leb = k % UInt32
    for i in (depth(k) - 1):-1:0
        m = (w1 .+ w3) ./ 2f0                      # the LONGEST edge, (v1, v3)
        w1, w2, w3 = ((leb >> i) & 1) == 0 ? (w1, m, w2) : (w2, m, w3)
    end
    mix(w) = Vec2f(w[1]*c1[1] + w[2]*c2[1] + w[3]*c3[1],
                   w[1]*c1[2] + w[2]*c2[2] + w[3]*c3[2])
    return (mix(w1), mix(w2), mix(w3))
end

# `emit!` is one call on the slide and three GPU passes in the demo: classify
# writes a count per key, an exclusive scan turns the counts into offsets, and
# scatter writes each thread's keys at its own offset. A thread cannot push.
emit!(out::Keys, ks::Key...) = (append!(out.v, ks); nothing)
