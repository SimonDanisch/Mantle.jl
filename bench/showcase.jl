# A crystal field at dusk: deferred shading, GPU culling, GGX materials, a
# directional shadow map, ambient occlusion and bloom. Twenty passes.
#
#   clear ─┬─> cull ─────┬─> draw count ─┬─> shadow ─> read ─────┐
#   lights ┘   sun cull ─┘               └─> gbuffer ─> read x4 ─┴─┬─> tiles ────┐
#                                                                  └─> ssao ─> ao blur ─┐
#                                                                                       │
#           composite <─ blur y <─ blur x <─ bright <─ light <────────────────────────── ┘
#
# The scene is one instanced prototype drawn once. The last NLIGHT instances are
# the point lights themselves, so the sources are visible and the compute pass
# that moves them moves their geometry too.
#
# Culling happens twice with one kernel: against the camera for what is drawn,
# and against the sun for what casts. Each writes its own draw command, so the
# host never learns either count.
#
# Lighting writes HDR to a buffer rather than to the screen, because the bloom
# chain has to read it back and a colour attachment would have to be copied out
# first. The composite is the pass that reaches the window.

import Mantle
using Lava, GeometryBasics, LinearAlgebra, KernelAbstractions, Atomix, Random
using ColorTypes: RGBA
using ColorTypes.FixedPointNumbers: N0f8
M = Mantle

W, H = 1600, 900
NPX = W * H
NSHARD = 5000
NLIGHT = 44
NINST = NSHARD + NLIGHT
VPB = 72     # vertices per shard
TILE = 16
NTX, NTY = cld(W, TILE), cld(H, TILE)
NTILE = NTX * NTY
PERTILE = 32
BSHIFT = 4                       # bloom runs at a quarter of each axis
BW, BH = cld(W, BSHIFT), cld(H, BSHIFT)
SMAP = 2048                      # shadow map edge
SEXTENT = 150f0                  # half-width of the box the sun sees
SDEPTH = 520f0
AOW, AOH = cld(W, 2), cld(H, 2)  # occlusion runs at half of each axis
AOSAMPLES = 12

# a six-sided crystal: a base ring, a narrower ring, and an apex
function shard_geometry()
    pos, nrm = Vec4f[], Vec4f[]
    n = 6
    ring(y, r) = [Vec3f(r * cos(2f0 * pi * i / n), y, r * sin(2f0 * pi * i / n)) for i in 0:(n - 1)]
    bot, top = ring(-1f0, 1f0), ring(0.45f0, 0.72f0)
    apex, base = Vec3f(0, 1, 0), Vec3f(0, -1, 0)
    tri!(a, b, c) = begin
        fn = normalize(cross(b - a, c - a))
        for p in (a, b, c)
            push!(pos, Vec4f(p..., 1)); push!(nrm, Vec4f(fn..., 0))
        end
    end
    for i in 1:n
        j = mod1(i + 1, n)
        tri!(bot[i], top[j], bot[j]); tri!(bot[i], top[i], top[j])
        tri!(top[i], apex, top[j])
        tri!(base, bot[j], bot[i])
    end
    pos, nrm
end

# clustered rather than uniform, so the field has groves and clearings
# An open plaza ringed by tall crystals, with a field of smaller ones beyond it:
# somewhere to stand, something to look at, and something behind that.
function crystals(n; clearing = 46f0, nhero = 26, nseed = 34, spread = 24f0,
                   extent = 230f0, hmax = 13f0, fat = 0.30f0)
    centers, sizes, colors, params = Vec4f[], Vec4f[], Vec4f[], Vec4f[]
    # the floor, a wide flat shard, dark and glossy so the lights streak on it
    push!(centers, Vec4f(0, -0.02, 0, 900))
    push!(sizes, Vec4f(900, 0.02, 900, 0))
    push!(colors, Vec4f(0.045, 0.047, 0.056, 0.13))
    push!(params, Vec4f(0.30, 0, 0, 0))
    one!(x, z, hgt, rad) = begin
        push!(centers, Vec4f(x, hgt, z, sqrt(rad^2 + hgt^2) * 1.05f0))
        push!(sizes, Vec4f(rad, hgt, rad, 2f0 * pi * rand(Float32)))
        # mostly cold, a few warm, and roughness skewed towards glossy
        t = rand(Float32)
        c = t < 0.80f0 ? Vec3f(0.26 + 0.16t, 0.32 + 0.14t, 0.46 + 0.12t) :
                         Vec3f(0.62, 0.44, 0.30) * (0.7f0 + 0.5f0 * rand(Float32))
        push!(colors, Vec4f(c[1], c[2], c[3], 0.08f0 + 0.42f0 * rand(Float32)^2))
        push!(params, Vec4f(rand(Float32) < 0.4f0 ? 0.9f0 : 0.05f0, 0, 0, 0))
    end
    for _ in 1:nhero                      # the ring, for silhouette and scale
        a = 2pi * rand(Float32)
        r = clearing + 18f0 * rand(Float32)
        hgt = 20f0 + 22f0 * rand(Float32)
        one!(r * cos(a), r * sin(a), hgt, 0.20f0 * hgt * (0.7f0 + 0.5f0 * rand(Float32)))
    end
    seeds = filter(p -> hypot(p...) > clearing + 12,
                   [Vec2f(extent * (rand(Float32) - 0.5f0), extent * (rand(Float32) - 0.5f0))
                    for _ in 1:(3nseed)])[1:nseed]
    while length(centers) < n
        s = seeds[rand(1:nseed)]
        x = s[1] + spread * randn(Float32)
        z = s[2] + spread * randn(Float32)
        hypot(x, z) < clearing && continue
        hgt = 1.5f0 + hmax * rand(Float32)^2.2f0
        one!(x, z, hgt, fat * hgt * (0.55f0 + 0.9f0 * rand(Float32)))
    end
    centers, sizes, colors, params
end

function tint(sat)
    h = 6rand(Float32); i = floor(Int, h); f = h - i
    c = (Vec3f(1, f, 0), Vec3f(1 - f, 1, 0), Vec3f(0, 1, f),
         Vec3f(0, 1 - f, 1), Vec3f(f, 0, 1), Vec3f(1, 0, 1 - f))[i + 1]
    Vec3f(1 - sat) + sat * c
end

function camera(t, radius, height)
    eye = Vec3f(radius * cos(0.11f0 * t), height + 4f0 * sin(0.17f0 * t), radius * sin(0.11f0 * t))
    at = Vec3f(0, 13, 0)
    f = normalize(at - eye); r = normalize(cross(f, Vec3f(0, 1, 0))); u = cross(r, f)
    view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                 -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
    tn = tan(50f0 * Float32(pi) / 360f0)
    near, far = 0.5f0, 400f0
    proj = Mat4f(1 / (Float32(W / H) * tn), 0, 0, 0, 0, -1 / tn, 0, 0,
                 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0)
    proj * view, eye
end

# An orthographic box looking down the sun's direction, wide enough to hold the
# part of the field worth shadowing. Everything outside it is left unshadowed
# rather than clamped, which is a lie that shows up as a bar of shade at the edge.
function sunmatrix(sundir::Vec3f, extent, depth)
    f = -normalize(sundir)                     # the way the light travels
    up = abs(f[2]) > 0.95f0 ? Vec3f(1, 0, 0) : Vec3f(0, 1, 0)
    r = normalize(cross(f, up)); u = cross(r, f)
    eye = normalize(sundir) * (0.5f0 * depth)
    view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                 -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
    near, far = 1f0, depth
    proj = Mat4f(1 / extent, 0, 0, 0, 0, -1 / extent, 0, 0,
                 0, 0, -1 / (far - near), 0, 0, 0, -near / (far - near), 1)
    proj * view
end

@kernel function reset_counter!(counter)
    counter[1] = UInt32(0)
end

@inline function frustum_plane(vp::Mat4f, k::Int32)
    row(i) = Vec4f(vp[i, 1], vp[i, 2], vp[i, 3], vp[i, 4])
    w = row(4)
    p = k == Int32(1) ? w + row(1) : k == Int32(2) ? w - row(1) :
        k == Int32(3) ? w + row(2) : k == Int32(4) ? w - row(2) :
        k == Int32(5) ? row(3) : w - row(3)
    p / sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
end

# `slot` comes off an atomic, so it is a number the GPU chose and the CPU cannot
# check. Bounding it here is not defensive programming — an unbounded index into
# a device buffer is a write into whatever the allocator put next, and the pool
# puts other buffers there. The symptom is that a *different* buffer quietly
# fills with ascending instance indices some minutes into a run.
@kernel function cull!(visible, counter, @Const(centers), vp::Mat4f, n::UInt32)
    i = @index(Global)
    @inbounds begin
        c = centers[i]
        keep = true
        for k in Int32(1):Int32(6)
            p = frustum_plane(vp, k)
            keep &= (p[1] * c[1] + p[2] * c[2] + p[3] * c[3] + p[4]) >= -c[4]
        end
        if keep
            slot = Atomix.@atomic counter[1] += UInt32(1)
            if slot <= n
                visible[slot] = UInt32(i)
            end
        end
    end
end

@kernel function write_draw!(cmds, @Const(counter), vpb::UInt32, n::UInt32)
    @inbounds cmds[1] = DrawIndirectCommand(min(counter[1], n) * vpb, UInt32(1), UInt32(0), UInt32(0))
end

# Moves the lights and the crystals that stand for them: the last NLIGHT
# instances are the sources, so what is lit and what is visible cannot disagree.
@kernel function move_lights!(pos, centers, params, @Const(home), @Const(lcol),
                              t::Float32, first::Int32)
    i = @index(Global)
    @inbounds begin
        h = home[i]
        ph = 0.7f0 * Float32(i)
        p = Vec4f(h[1] + 7f0 * sin(0.42f0 * t + ph), h[2] + 1.6f0 * sin(0.9f0 * t + ph),
                  h[3] + 7f0 * cos(0.33f0 * t + ph), h[4])
        pos[i] = p
        c = lcol[i]
        centers[first + i] = Vec4f(p[1], p[2], p[3], centers[first + i][4])
        params[first + i] = Vec4f(0f0, c[4], 0f0, 0f0)     # emissive strength
    end
end

function gbuffer_vertex(protopos, protonrm, visible, centers, sizes, colors, params, vp::Mat4f)
    v = vertex_index() - Int32(1)
    inst = v ÷ Int32(72)
    k = v - inst * Int32(72)
    @inbounds begin
        o = Int32(visible[inst + Int32(1)])
        c = centers[o]; s = sizes[o]
        lp = protopos[k + Int32(1)]; ln = protonrm[k + Int32(1)]
        ca, sa = cos(s[4]), sin(s[4])
        px, py, pz = lp[1] * s[1], lp[2] * s[2], lp[3] * s[3]
        # the normal of a non-uniformly scaled surface is the inverse scale, and
        # a yaw is its own inverse transpose
        nx, ny, nz = ln[1] / s[1], ln[2] / s[2], ln[3] / s[3]
        world = Vec4f(c[1] + ca * px + sa * pz, c[2] + py, c[3] - sa * px + ca * pz, 1f0)
        n = normalize(Vec3f(ca * nx + sa * nz, ny, -sa * nx + ca * nz))
        (position = vp * world, albedo = colors[o], surface = params[o],
         normal = n, wpos = Vec3f(world[1], world[2], world[3]))
    end
end

# albedo and roughness, then metalness and emission, then the normal: what a
# fragment knows about a surface, split so a copy of each is one UInt32.
function gbuffer_fragment(inputs)
    a = inputs.albedo
    s = inputs.surface
    n = normalize(inputs.normal)
    (Vec4f(a[1], a[2], a[3], a[4]),
     Vec4f(s[1], min(1f0, s[2]), 0f0, 1f0),
     Vec4f(0.5f0 * n[1] + 0.5f0, 0.5f0 * n[2] + 0.5f0, 0.5f0 * n[3] + 0.5f0, 1f0))
end

GBUFFER = Rasterizer(vertex = gbuffer_vertex, fragment = gbuffer_fragment,
                     varyings = (albedo = Vec4f, surface = Vec4f, normal = Vec3f, wpos = Vec3f),
                     topology = TriangleList(), blend = Opaque(),
                     cull = CullBack(), depth = DepthLess())

# The same instancing as the g-buffer pass with nothing but the position kept:
# a depth-only pass has no attachment to write, so its fragment returns nothing.
function shadow_vertex(protopos, visible, centers, sizes, vp::Mat4f)
    v = vertex_index() - Int32(1)
    inst = v ÷ Int32(72)
    k = v - inst * Int32(72)
    @inbounds begin
        o = Int32(visible[inst + Int32(1)])
        c = centers[o]; s = sizes[o]
        lp = protopos[k + Int32(1)]
        ca, sa = cos(s[4]), sin(s[4])
        px, py, pz = lp[1] * s[1], lp[2] * s[2], lp[3] * s[3]
        (position = vp * Vec4f(c[1] + ca * px + sa * pz, c[2] + py,
                               c[3] - sa * px + ca * pz, 1f0),)
    end
end
shadow_fragment(inputs) = nothing

SHADOW = Rasterizer(vertex = shadow_vertex, fragment = shadow_fragment,
                    varyings = NamedTuple(), topology = TriangleList(),
                    blend = Opaque(), cull = CullBack(), depth = DepthLess())

@inline unpack8(px::UInt32, shift::UInt32) = Float32((px >> shift) & 0x000000ff) * (1f0 / 255f0)

# Three by three, so the edge of a shadow is a gradient over a few texels rather
# than a staircase.
#
# Both biases are slope scaled. A surface nearly edge-on to the sun crosses many
# texels' worth of depth within one texel, so a constant offset that is enough
# there detaches the shadow everywhere else, and one that looks right head-on
# leaves the grazing faces striped with their own shadow. `texel` is what a
# shadow texel measures in world units, which is the unit both corrections are
# naturally in.
@inline function sunshadow(smap, sunvp::Mat4f, world::Vec3f, n::Vec3f, ndl::Float32,
                           res::Int32, texel::Float32, bias::Float32)
    slope = clamp(sqrt(max(0f0, 1f0 - ndl * ndl)) / max(ndl, 0.08f0), 0f0, 10f0)
    w = world + n * (texel * (1.6f0 + 2.2f0 * slope))
    q = sunvp * Vec4f(w[1], w[2], w[3], 1f0)
    (abs(q[1]) > 1f0 || abs(q[2]) > 1f0 || q[3] < 0f0 || q[3] > 1f0) && return 1f0
    fx = (0.5f0 * q[1] + 0.5f0) * Float32(res)
    fy = (0.5f0 * q[2] + 0.5f0) * Float32(res)
    ref = q[3] - bias * (1f0 + slope)
    lit = 0f0
    @inbounds for dy in Int32(-1):Int32(1), dx in Int32(-1):Int32(1)
        sx = clamp(unsafe_trunc(Int32, fx) + dx, Int32(0), res - Int32(1))
        sy = clamp(unsafe_trunc(Int32, fy) + dy, Int32(0), res - Int32(1))
        lit += smap[sy * res + sx + Int32(1)] > ref ? 1f0 : 0f0
    end
    lit * (1f0 / 9f0)
end

@inline function unproject(invvp::Mat4f, fx::Float32, fy::Float32, d::Float32, w::Int32, h::Int32)
    q = invvp * Vec4f(2f0 * fx / Float32(w) - 1f0, 2f0 * fy / Float32(h) - 1f0, d, 1f0)
    Vec3f(q[1] / q[4], q[2] / q[4], q[3] / q[4])
end

# dusk: warm near the horizon, deep overhead, and a glow where the sun is
@inline function sky(dir::Vec3f, sundir::Vec3f, suncol::Vec3f)
    t = clamp(0.5f0 * (dir[2] + 1f0), 0f0, 1f0)
    horizon = Vec3f(0.115f0, 0.085f0, 0.105f0)
    zenith = Vec3f(0.012f0, 0.020f0, 0.055f0)
    base = horizon + (zenith - horizon) * t^0.55f0
    c = max(0f0, dir[1] * sundir[1] + dir[2] * sundir[2] + dir[3] * sundir[3])
    base + suncol * (c^220f0 * 7f0 + c^12f0 * 0.10f0)
end

# Cook-Torrance with GGX, which is what makes a rough surface look rough and a
# metal look like metal rather than like plastic with a bright spot on it.
@inline function ggx(n::Vec3f, v::Vec3f, l::Vec3f, albedo::Vec3f, rough::Float32, metal::Float32)
    ndl = n[1] * l[1] + n[2] * l[2] + n[3] * l[3]
    ndv = n[1] * v[1] + n[2] * v[2] + n[3] * v[3]
    (ndl <= 0f0 || ndv <= 0f0) && return Vec3f(0)
    h = normalize(v + l)
    ndh = max(0f0, n[1] * h[1] + n[2] * h[2] + n[3] * h[3])
    vdh = max(0f0, v[1] * h[1] + v[2] * h[2] + v[3] * h[3])
    a = max(1f-3, rough * rough)
    a2 = a * a
    dd = ndh * ndh * (a2 - 1f0) + 1f0
    D = a2 / (Float32(pi) * dd * dd)
    k = a * 0.5f0
    G = (ndl / (ndl * (1f0 - k) + k)) * (ndv / (ndv * (1f0 - k) + k))
    f0 = Vec3f(0.04f0) + (albedo - Vec3f(0.04f0)) * metal
    fr = (1f0 - vdh)^5
    F = f0 + (Vec3f(1f0) - f0) * fr
    spec = F * (D * G / max(1f-4, 4f0 * ndl * ndv))
    kd = (Vec3f(1f0) - F) * (1f0 - metal)
    (kd * albedo * (1f0 / Float32(pi)) + spec) * ndl
end

@kernel function tile_lights!(tilelights, tilecount, @Const(dep), @Const(lpos), invvp::Mat4f,
                              w::Int32, h::Int32, tile::Int32, ntx::Int32,
                              nlight::Int32, pertile::Int32)
    tx0, ty0 = @index(Global, NTuple)
    @inbounds begin
        ti = (Int32(ty0) - Int32(1)) * ntx + Int32(tx0)
        x0 = (Int32(tx0) - Int32(1)) * tile; y0 = (Int32(ty0) - Int32(1)) * tile
        x1 = min(x0 + tile, w); y1 = min(y0 + tile, h)
        dmin = 1f0; dmax = 0f0
        for py in y0:(y1 - Int32(1)), px in x0:(x1 - Int32(1))
            d = dep[py * w + px + Int32(1)]
            if d < 1f0
                dmin = min(dmin, d); dmax = max(dmax, d)
            end
        end
        n = UInt32(0)
        if dmax > 0f0
            lo = Vec3f(1f30, 1f30, 1f30)
            hi = Vec3f(-1f30, -1f30, -1f30)
            for c in Int32(0):Int32(7)
                p = unproject(invvp, Float32((c & Int32(1)) == 0 ? x0 : x1),
                              Float32((c & Int32(2)) == 0 ? y0 : y1),
                              (c & Int32(4)) == 0 ? dmin : dmax, w, h)
                lo = Vec3f(min(lo[1], p[1]), min(lo[2], p[2]), min(lo[3], p[3]))
                hi = Vec3f(max(hi[1], p[1]), max(hi[2], p[2]), max(hi[3], p[3]))
            end
            for l in Int32(1):nlight
                lp = lpos[l]
                dx = max(0f0, max(lo[1] - lp[1], lp[1] - hi[1]))
                dy = max(0f0, max(lo[2] - lp[2], lp[2] - hi[2]))
                dz = max(0f0, max(lo[3] - lp[3], lp[3] - hi[3]))
                if dx * dx + dy * dy + dz * dz <= lp[4] * lp[4] && n < UInt32(pertile)
                    n += UInt32(1)
                    tilelights[(ti - Int32(1)) * pertile + Int32(n)] = UInt32(l)
                end
            end
        end
        tilecount[ti] = n
    end
end

# A workgroup is exactly one light tile, which is also the shape that keeps the
# g-buffer read and the HDR write coalesced.
@kernel function shade!(hdr, @Const(alb), @Const(mat), @Const(nrm), @Const(dep),
                        @Const(lpos), @Const(lcol), @Const(tilelights), @Const(tilecount),
                        @Const(smap), @Const(aomap), invvp::Mat4f, eye::Vec4f,
                        sun::Vec4f, suncol::Vec4f, sunvp::Mat4f,
                        w::Int32, h::Int32, tile::Int32, ntx::Int32,
                        pertile::Int32, fog::Float32, emitscale::Float32,
                        smapres::Int32, stexel::Float32, sbias::Float32,
                        aow::Int32, aoh::Int32, aoamb::Float32, aodirect::Float32)
    ix, iy = @index(Global, NTuple)
    px = Int32(ix) - Int32(1)
    py = Int32(iy) - Int32(1)
    @inbounds begin
        i = py * w + px + Int32(1)
        d = dep[i]
        sundir = Vec3f(sun[1], sun[2], sun[3])
        sunrgb = Vec3f(suncol[1], suncol[2], suncol[3]) * suncol[4]
        world = unproject(invvp, Float32(px) + 0.5f0, Float32(py) + 0.5f0, min(d, 0.999999f0), w, h)
        eyep = Vec3f(eye[1], eye[2], eye[3])
        ray = normalize(world - eyep)
        out = sky(ray, sundir, sunrgb)
        if d < 1f0
            a = alb[i]; m = mat[i]; nn = nrm[i]
            albedo = Vec3f(unpack8(a, UInt32(0)), unpack8(a, UInt32(8)), unpack8(a, UInt32(16)))
            rough = max(0.04f0, unpack8(a, UInt32(24)))
            metal = unpack8(m, UInt32(0))
            emit = unpack8(m, UInt32(8))
            n = normalize(Vec3f(2f0 * unpack8(nn, UInt32(0)) - 1f0,
                                2f0 * unpack8(nn, UInt32(8)) - 1f0,
                                2f0 * unpack8(nn, UInt32(16)) - 1f0))
            view = Vec3f(-ray[1], -ray[2], -ray[3])
            # a hemisphere of sky above and bounce from the ground below, which is
            # what keeps the unlit side from being flat black
            # bilinear off the half-resolution map, so the upsample does not
            # print the occlusion grid onto every surface
            fax = (Float32(px) + 0.5f0) * 0.5f0 - 0.5f0
            fay = (Float32(py) + 0.5f0) * 0.5f0 - 0.5f0
            ax0 = clamp(unsafe_trunc(Int32, fax), Int32(0), aow - Int32(1))
            ay0 = clamp(unsafe_trunc(Int32, fay), Int32(0), aoh - Int32(1))
            ax1 = min(ax0 + Int32(1), aow - Int32(1)); ay1 = min(ay0 + Int32(1), aoh - Int32(1))
            atx = clamp(fax - Float32(ax0), 0f0, 1f0); aty = clamp(fay - Float32(ay0), 0f0, 1f0)
            a00 = aomap[ay0 * aow + ax0 + Int32(1)]; a10 = aomap[ay0 * aow + ax1 + Int32(1)]
            a01 = aomap[ay1 * aow + ax0 + Int32(1)]; a11 = aomap[ay1 * aow + ax1 + Int32(1)]
            occ = (a00 + (a10 - a00) * atx) + ((a01 + (a11 - a01) * atx) -
                  (a00 + (a10 - a00) * atx)) * aty
            # Full strength on the ambient, which is the term it actually models,
            # and a share of the point lights, which is not physical but is what
            # keeps a crevice from being filled in by whatever drifts past it.
            ao = clamp(1f0 - occ * aoamb, 0f0, 1f0)
            aol = 1f0 - aodirect * (1f0 - ao)
            up = 0.5f0 * n[2] + 0.5f0
            amb = (Vec3f(0.055f0, 0.070f0, 0.115f0) * up +
                   Vec3f(0.020f0, 0.017f0, 0.016f0) * (1f0 - up)) * (1f0 - 0.7f0 * metal) * ao
            ndl = max(0f0, n[1] * sundir[1] + n[2] * sundir[2] + n[3] * sundir[3])
            sh = ndl > 0f0 ? sunshadow(smap, sunvp, world, n, ndl, smapres, stexel, sbias) : 1f0
            acc = albedo * amb + ggx(n, view, sundir, albedo, rough, metal) * sunrgb * sh
            ti = (py ÷ tile) * ntx + (px ÷ tile) + Int32(1)
            for k in Int32(1):Int32(tilecount[ti])
                l = Int32(tilelights[(ti - Int32(1)) * pertile + k])
                lp = lpos[l]
                dv = Vec3f(lp[1] - world[1], lp[2] - world[2], lp[3] - world[3])
                dist = sqrt(dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3]) + 1f-4
                fall = max(0f0, 1f0 - dist / lp[4])
                if fall > 0f0
                    c = lcol[l]
                    acc += ggx(n, view, dv / dist, albedo, rough, metal) *
                           (Vec3f(c[1], c[2], c[3]) * (c[4] * fall * fall * aol))
                end
            end
            # the sources themselves, so the lights are things and not just effects
            acc += albedo * (emit * emitscale)
            dist = sqrt((world[1] - eyep[1])^2 + (world[2] - eyep[2])^2 + (world[3] - eyep[3])^2)
            f = 1f0 - exp(-dist * fog)
            out = acc + (sky(ray, sundir, sunrgb) - acc) * f
        end
        # no tone map here: bloom wants the values before they are squashed
        hdr[i] = Vec4f(out[1], out[2], out[3], 1f0)
    end
end

@inline function hash01(x::Int32, y::Int32)
    h = UInt32(x) * 0x9e3779b9 + UInt32(y) * 0x85ebca6b
    h = h ⊻ (h >> 15)
    h = h * 0x2545f491
    Float32(h & 0x00ffffff) * (1f0 / 1.6777216f7)
end

# Occlusion the way a depth buffer can answer it: put a cosine-weighted
# hemisphere of points around the surface, ask what the g-buffer actually holds
# where each one lands, and count the ones that turn out to be in front.
#
# Half resolution, because the samples are scattered reads into the depth buffer
# and there are AOSAMPLES of them per pixel. The blur below is what makes half
# resolution and twelve samples look like more of both.
@kernel function ssao!(ao, @Const(dep), @Const(nrm), vp::Mat4f, invvp::Mat4f, eye::Vec4f,
                       w::Int32, h::Int32, aow::Int32, aoh::Int32,
                       radius::Float32, bias::Float32, nsample::Int32)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        px = (Int32(ix) - Int32(1)) * Int32(2)
        py = (Int32(iy) - Int32(1)) * Int32(2)
        i = py * w + px + Int32(1)
        d = dep[i]
        occ = 0f0
        if d < 1f0
            nn = nrm[i]
            n = normalize(Vec3f(2f0 * unpack8(nn, UInt32(0)) - 1f0,
                                2f0 * unpack8(nn, UInt32(8)) - 1f0,
                                2f0 * unpack8(nn, UInt32(16)) - 1f0))
            p = unproject(invvp, Float32(px) + 0.5f0, Float32(py) + 0.5f0, d, w, h)
            ep = Vec3f(eye[1], eye[2], eye[3])
            up = abs(n[3]) < 0.9f0 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0)
            t = normalize(cross(up, n)); b = cross(n, t)
            rot = hash01(px, py) * 6.2831854f0
            for k in Int32(0):(nsample - Int32(1))
                a = (Float32(k) + 0.5f0) / Float32(nsample)
                rr = sqrt(a)
                phi = Float32(k) * 2.3999632f0 + rot
                dir = t * (rr * cos(phi)) + b * (rr * sin(phi)) + n * sqrt(max(0f0, 1f0 - a))
                s = p + dir * (radius * (0.3f0 + 0.7f0 * a))
                q = vp * Vec4f(s[1], s[2], s[3], 1f0)
                if q[4] > 0f0
                    sx = (0.5f0 * q[1] / q[4] + 0.5f0) * Float32(w)
                    sy = (0.5f0 * q[2] / q[4] + 0.5f0) * Float32(h)
                    if sx >= 0f0 && sx < Float32(w) && sy >= 0f0 && sy < Float32(h)
                        jx = unsafe_trunc(Int32, sx); jy = unsafe_trunc(Int32, sy)
                        ds = dep[jy * w + jx + Int32(1)]
                        if ds < 1f0
                            ps = unproject(invvp, sx, sy, ds, w, h)
                            dp = sqrt((ps[1] - ep[1])^2 + (ps[2] - ep[2])^2 + (ps[3] - ep[3])^2)
                            dsamp = sqrt((s[1] - ep[1])^2 + (s[2] - ep[2])^2 + (s[3] - ep[3])^2)
                            if dp < dsamp - bias
                                # a surface far behind the point is a different
                                # object, not a crevice, so it must not darken it
                                gap = sqrt((ps[1] - p[1])^2 + (ps[2] - p[2])^2 + (ps[3] - p[3])^2)
                                occ += clamp(radius / max(1f-4, gap), 0f0, 1f0)
                            end
                        end
                    end
                end
            end
        end
        # The depth rides along so the blur reads one contiguous half resolution
        # buffer instead of twenty-five scattered full resolution depth fetches
        # per pixel. Worth about 0.015 ms here, which is less than it sounds like
        # it should be: this blur costs 0.078 ms measured on its own, and what
        # `timings` reports for the pass is mostly waiting for `ssao` to drain.
        ao[(Int32(iy) - Int32(1)) * aow + Int32(ix)] = Vec2f(occ / Float32(nsample), d)
    end
end

# Bilateral: a box blur alone drags occlusion across a silhouette, which is a
# dark halo around every crystal. The depth term is what keeps it on its side.
@kernel function ao_blur!(dst, @Const(src), aow::Int32, aoh::Int32, sharp::Float32)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        d0 = src[(Int32(iy) - Int32(1)) * aow + Int32(ix)][2]
        acc = 0f0; wsum = 0f0
        for dy in Int32(-2):Int32(2), dx in Int32(-2):Int32(2)
            sx = clamp(Int32(ix) + dx, Int32(1), aow)
            sy = clamp(Int32(iy) + dy, Int32(1), aoh)
            v = src[(sy - Int32(1)) * aow + sx]
            g = exp(-abs(v[2] - d0) * sharp)
            acc += v[1] * g
            wsum += g
        end
        dst[(Int32(iy) - Int32(1)) * aow + Int32(ix)] = acc / wsum
    end
end

# Everything above the threshold, downsampled by BSHIFT on the way.
#
# Four taps across the block rather than all sixteen. A lane's taps are strided
# by `shift`, so each one is its own cache line however many it takes, and the
# blur that follows is wider than the block: reading every pixel cost 0.59 ms
# against 0.15 and there is nothing in the result to show for it.
@kernel function bright!(dst, @Const(hdr), w::Int32, h::Int32, bw::Int32,
                         shift::Int32, thr::Float32)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        acc = Vec3f(0)
        half = shift ÷ Int32(2)
        for dy in (Int32(0), half), dx in (Int32(0), half)
            sx = min((Int32(ix) - Int32(1)) * shift + dx, w - Int32(1))
            sy = min((Int32(iy) - Int32(1)) * shift + dy, h - Int32(1))
            c = hdr[sy * w + sx + Int32(1)]
            m = max(c[1], max(c[2], c[3]))
            acc += Vec3f(c[1], c[2], c[3]) * (m > thr ? (m - thr) / m : 0f0)
        end
        dst[(Int32(iy) - Int32(1)) * bw + Int32(ix)] =
            Vec4f(acc[1] * 0.25f0, acc[2] * 0.25f0, acc[3] * 0.25f0, 1f0)
    end
end

# Separable, so two passes of thirteen taps rather than one of a hundred and sixty-nine.
@kernel function blur!(dst, @Const(src), bw::Int32, bh::Int32, dx::Int32, dy::Int32)
    ix, iy = @index(Global, NTuple)
    @inbounds begin
        acc = Vec3f(0); wsum = 0f0
        for k in Int32(-6):Int32(6)
            sx = clamp(Int32(ix) + k * dx, Int32(1), bw)
            sy = clamp(Int32(iy) + k * dy, Int32(1), bh)
            c = src[(sy - Int32(1)) * bw + sx]
            g = exp(-Float32(k * k) * (1f0 / 16f0))
            acc += Vec3f(c[1], c[2], c[3]) * g
            wsum += g
        end
        dst[(Int32(iy) - Int32(1)) * bw + Int32(ix)] =
            Vec4f(acc[1] / wsum, acc[2] / wsum, acc[3] / wsum, 1f0)
    end
end

function composite_vertex()
    v = vertex_index()
    p = v == Int32(1) ? Vec4f(-1, -1, 0, 1) : v == Int32(2) ? Vec4f(3, -1, 0, 1) : Vec4f(-1, 3, 0, 1)
    (position = p,)
end

# Bilinear on the way back up: nearest across a quarter-resolution blur is a
# grid of squares wherever the bloom has an edge, which is everywhere it matters.
function composite_fragment(inputs, hdr, bloom, w::Int32, h::Int32, bw::Int32, bh::Int32,
                            shift::Int32, exposure::Float32, amount::Float32)
    px = unsafe_trunc(Int32, frag_coord_x())
    py = unsafe_trunc(Int32, frag_coord_y())
    @inbounds begin
        c = hdr[py * w + px + Int32(1)]
        fx = (Float32(px) + 0.5f0) / Float32(shift) - 0.5f0
        fy = (Float32(py) + 0.5f0) / Float32(shift) - 0.5f0
        x0 = clamp(unsafe_trunc(Int32, fx), Int32(0), bw - Int32(1))
        y0 = clamp(unsafe_trunc(Int32, fy), Int32(0), bh - Int32(1))
        x1 = min(x0 + Int32(1), bw - Int32(1))
        y1 = min(y0 + Int32(1), bh - Int32(1))
        tx = clamp(fx - Float32(x0), 0f0, 1f0)
        ty = clamp(fy - Float32(y0), 0f0, 1f0)
        b00 = bloom[y0 * bw + x0 + Int32(1)]; b10 = bloom[y0 * bw + x1 + Int32(1)]
        b01 = bloom[y1 * bw + x0 + Int32(1)]; b11 = bloom[y1 * bw + x1 + Int32(1)]
        bt = b00 + (b10 - b00) * tx
        bb = b01 + (b11 - b01) * tx
        b = bt + (bb - bt) * ty
        v = Vec3f(c[1] + b[1] * amount, c[2] + b[2] * amount, c[3] + b[3] * amount) * exposure
        Vec4f(v[1] / (1f0 + v[1]), v[2] / (1f0 + v[2]), v[3] / (1f0 + v[3]), 1f0)
    end
end

COMPOSITE = Rasterizer(vertex = composite_vertex, fragment = composite_fragment,
                       varyings = NamedTuple(), topology = TriangleList(),
                       blend = Opaque(), cull = NoCull(), depth = DepthOff())

Random.seed!(9)
protopos, protonrm = shard_geometry()
boxc, boxs, boxcol, boxpar = crystals(NSHARD)
lighthue = [Vec4f(tint(0.9f0)..., 5.5f0) for _ in 1:NLIGHT]
for i in 1:NLIGHT                          # the sources ride along as instances
    r = 0.4f0 + 0.7f0 * rand(Float32)
    push!(boxc, Vec4f(0, 0, 0, 2r))
    push!(boxs, Vec4f(r, r * 1.6f0, r, 2f0 * pi * rand(Float32)))
    push!(boxcol, Vec4f(lighthue[i][1], lighthue[i][2], lighthue[i][3], 0.5))
    push!(boxpar, Vec4f(0, 1, 0, 0))
end

dev = M.Device(Lava)
win = M.Window(W, H; title = "mantle: crystal field", vsync = false)

pp = M.Buffer(dev, protopos)
pn = M.Buffer(dev, protonrm)
centers = M.Buffer(dev, boxc)
sizes   = M.Buffer(dev, boxs)
colors  = M.Buffer(dev, boxcol)
params  = M.Buffer(dev, boxpar)
# ones, not undef: read as an index the moment anything reads a stale slot
visible = M.Buffer(dev, fill(UInt32(1), NINST))
counter = M.Buffer(dev, UInt32[0])
drawcmd = M.Buffer(dev, DrawIndirectCommand, 1)
# the same three again, for what the sun can see
sunvisible = M.Buffer(dev, fill(UInt32(1), NINST))
suncounter = M.Buffer(dev, UInt32[0])
sundrawcmd = M.Buffer(dev, DrawIndirectCommand, 1)

# Spread over the plaza and up among the ring, so some light the floor and some
# catch the crystals. A reach of 24 to 46 is what makes a light a thing in the
# scene rather than a dot with a puddle under it.
lighthome = M.Buffer(dev, [(a = 2pi * rand(Float32); r = 12f0 + 62f0 * rand(Float32);
                            Vec4f(r * cos(a), 2f0 + 17f0 * rand(Float32)^1.6f0, r * sin(a),
                                  24f0 + 22f0 * rand(Float32))) for _ in 1:NLIGHT])
lightpos = M.Buffer(dev, zeros(Vec4f, NLIGHT))
lightcol = M.Buffer(dev, lighthue)

vpref = Ref(first(camera(0f0, 62f0, 14f0)))
cullref = Ref(vpref[])
invref = Ref(inv(vpref[]))
eyeref = Ref(Vec4f(0, 0, 0, 1))
timeref = Ref(0f0)
lightref = Ref(Int32(NLIGHT))
exposure = Ref(0.85f0)
fogref = Ref(0.0035f0)
threshold = Ref(1.6f0)
bloomamt = Ref(0.30f0)
emitref = Ref(9f0)
sunvpref = Ref(sunmatrix(Vec3f(-0.62, 0.045, 0.35), SEXTENT, SDEPTH))
sbiasref = Ref(0.004f0)
aoradius = Ref(4.5f0)
aobias = Ref(0.05f0)
aoamb = Ref(2.6f0)     # how hard occlusion presses on the ambient
aodirect = Ref(0.6f0)  # and how much of that reaches the point lights
sunref = Ref(Vec4f(normalize(Vec3f(-0.62, 0.22, 0.35))..., 0))
suncolref = Ref(Vec4f(1.0, 0.55, 0.28, 6.5))

graph = M.Graph(dev)
screen = M.Surface(graph, win)
albedo = M.Transient.Image(graph, RGBA{N0f8}, (W, H))
matter = M.Transient.Image(graph, RGBA{N0f8}, (W, H))
normal = M.Transient.Image(graph, RGBA{N0f8}, (W, H))
zbuf   = M.Transient.Image(graph, Float32, (W, H))
albedo_px = M.Transient.Buffer(graph, UInt32, NPX)
matter_px = M.Transient.Buffer(graph, UInt32, NPX)
normal_px = M.Transient.Buffer(graph, UInt32, NPX)
depth_px  = M.Transient.Buffer(graph, Float32, NPX)
tilelights = M.Transient.Buffer(graph, UInt32, NTILE * PERTILE)
tilecount  = M.Transient.Buffer(graph, UInt32, NTILE)
shadow    = M.Transient.Image(graph, Float32, (SMAP, SMAP))
shadow_px = M.Transient.Buffer(graph, Float32, SMAP * SMAP)
aoraw  = M.Transient.Buffer(graph, Vec2f, AOW * AOH)
aomap  = M.Transient.Buffer(graph, Float32, AOW * AOH)
hdr    = M.Transient.Buffer(graph, Vec4f, NPX)
bloomA = M.Transient.Buffer(graph, Vec4f, BW * BH)
bloomB = M.Transient.Buffer(graph, Vec4f, BW * BH)

M.compute!(graph, "clear count") do p
    M.dispatch!(p, reset_counter!, (M.use(p, counter; write = true),), 1)
    M.dispatch!(p, reset_counter!, (M.use(p, suncounter; write = true),), 1)
end
M.compute!(graph, "lights") do p
    M.dispatch!(p, move_lights!, (M.use(p, lightpos; write = true),
                                  M.use(p, centers; read = true, write = true),
                                  M.use(p, params; read = true, write = true),
                                  M.use(p, lighthome; read = true),
                                  M.use(p, lightcol; read = true),
                                  timeref, Int32(NSHARD)), NLIGHT)
end
M.compute!(graph, "cull") do p
    M.dispatch!(p, cull!, (M.use(p, visible; write = true),
                           M.use(p, counter; read = true, write = true),
                           M.use(p, centers; read = true), cullref, UInt32(NINST)), NINST)
end
M.compute!(graph, "sun cull") do p
    M.dispatch!(p, cull!, (M.use(p, sunvisible; write = true),
                           M.use(p, suncounter; read = true, write = true),
                           M.use(p, centers; read = true), sunvpref, UInt32(NINST)), NINST)
end
M.compute!(graph, "draw count") do p
    M.dispatch!(p, write_draw!, (M.use(p, drawcmd; write = true),
                                 M.use(p, counter; read = true), UInt32(VPB), UInt32(NINST)), 1)
    M.dispatch!(p, write_draw!, (M.use(p, sundrawcmd; write = true),
                                 M.use(p, suncounter; read = true), UInt32(VPB), UInt32(NINST)), 1)
end
M.render!(graph, "shadow", shadow => M.Clear(1f0)) do p
    args = (M.Attribute(p, pp), M.Attribute(p, sunvisible), M.Attribute(p, centers),
            M.Attribute(p, sizes), sunvpref)
    M.draw!(p, SHADOW, args, sundrawcmd)
end
M.copy!(graph, "read shadow", shadow_px, shadow)
M.render!(graph, "gbuffer", albedo => M.Clear((0f0, 0f0, 0f0, 1f0)),
                            matter => M.Discard, normal => M.Discard,
                            zbuf => M.Clear(1f0)) do p
    args = (M.Attribute(p, pp), M.Attribute(p, pn), M.Attribute(p, visible),
            M.Attribute(p, centers), M.Attribute(p, sizes), M.Attribute(p, colors),
            M.Attribute(p, params), vpref)
    M.draw!(p, GBUFFER, args, drawcmd)
end
M.copy!(graph, "read albedo", albedo_px, albedo)
M.copy!(graph, "read matter", matter_px, matter)
M.copy!(graph, "read normal", normal_px, normal)
M.copy!(graph, "read depth", depth_px, zbuf)
M.compute!(graph, "tiles") do p
    M.dispatch!(p, tile_lights!, (M.use(p, tilelights; write = true),
                                  M.use(p, tilecount; write = true),
                                  M.use(p, depth_px; read = true),
                                  M.use(p, lightpos; read = true), invref,
                                  Int32(W), Int32(H), Int32(TILE), Int32(NTX),
                                  lightref, Int32(PERTILE)), (NTX, NTY); group = (8, 8))
end
M.compute!(graph, "ssao") do p
    M.dispatch!(p, ssao!, (M.use(p, aoraw; write = true), M.use(p, depth_px; read = true),
                           M.use(p, normal_px; read = true), vpref, invref, eyeref,
                           Int32(W), Int32(H), Int32(AOW), Int32(AOH),
                           aoradius, aobias, Int32(AOSAMPLES)), (AOW, AOH); group = (16, 16))
end
M.compute!(graph, "ao blur") do p
    M.dispatch!(p, ao_blur!, (M.use(p, aomap; write = true), M.use(p, aoraw; read = true),
                              Int32(AOW), Int32(AOH), 900f0), (AOW, AOH); group = (16, 16))
end
M.compute!(graph, "light") do p
    args = (M.use(p, hdr; write = true),
            M.use(p, albedo_px; read = true), M.use(p, matter_px; read = true),
            M.use(p, normal_px; read = true), M.use(p, depth_px; read = true),
            M.use(p, lightpos; read = true), M.use(p, lightcol; read = true),
            M.use(p, tilelights; read = true), M.use(p, tilecount; read = true),
            M.use(p, shadow_px; read = true), M.use(p, aomap; read = true),
            invref, eyeref, sunref, suncolref, sunvpref,
            Int32(W), Int32(H), Int32(TILE), Int32(NTX), Int32(PERTILE), fogref, emitref,
            Int32(SMAP), 2f0 * SEXTENT / Float32(SMAP), sbiasref,
            Int32(AOW), Int32(AOH), aoamb, aodirect)
    M.dispatch!(p, shade!, args, (W, H); group = (TILE, TILE))
end
M.compute!(graph, "bright") do p
    M.dispatch!(p, bright!, (M.use(p, bloomA; write = true), M.use(p, hdr; read = true),
                             Int32(W), Int32(H), Int32(BW), Int32(BSHIFT), threshold),
                (BW, BH); group = (16, 16))
end
M.compute!(graph, "blur x") do p
    M.dispatch!(p, blur!, (M.use(p, bloomB; write = true), M.use(p, bloomA; read = true),
                           Int32(BW), Int32(BH), Int32(1), Int32(0)), (BW, BH); group = (16, 16))
end
M.compute!(graph, "blur y") do p
    M.dispatch!(p, blur!, (M.use(p, bloomA; write = true), M.use(p, bloomB; read = true),
                           Int32(BW), Int32(BH), Int32(0), Int32(1)), (BW, BH); group = (16, 16))
end
M.render!(graph, "composite", screen => M.Discard) do p
    M.draw!(p, COMPOSITE, (), 3;
            frag_args = (M.use(p, hdr; read = true), M.use(p, bloomA; read = true),
                         Int32(W), Int32(H), Int32(BW), Int32(BH), Int32(BSHIFT),
                         exposure, bloomamt))
end
plan = M.Plan(graph; profile = true)

function frame!(t, radius, height, freeze)
    vp, eye = camera(t, radius, height)
    vpref[] = vp
    # derived rather than set beside it: a shadow matrix that disagrees with the
    # sun direction is shadows falling the wrong way, and nothing says so
    s = sunref[]
    sunvpref[] = sunmatrix(Vec3f(s[1], s[2], s[3]), SEXTENT, SDEPTH)
    freeze || (cullref[] = vp)
    invref[] = inv(vp)
    eyeref[] = Vec4f(eye[1], eye[2], eye[3], 1f0)
    timeref[] = t
    M.run!(plan)
end

begin # ── the render loop ───────────────────────────────────────────────────────
    orbit = 40f0
    height = 16f0
    speed = 1f0
    freeze_cull = false
    stop = Threads.Atomic{Bool}(false)
    frames = Ref(0)
    nvisible = Ref(0)
    render = errormonitor(@async begin
        t, last = 0f0, time()
        while isopen(win) && !stop[]
            now = time(); t += speed * Float32(now - last); last = now
            frame!(t, orbit, height, freeze_cull)
            frames[] += 1
            frames[] % 30 == 0 && (nvisible[] = Int(Array(counter)[1]))
            sleep(0.001)
        end
    end)
end

# ── eval any of these while the window is up ─────────────────────────────────
#
# Every knob is followed by the line that puts it back, and those lines hold the
# defaults above rather than whatever they happened to be while this was written.

nvisible[]               # instances drawn, of NINST

sbiasref[] = 10f0        # every depth test passes: the same frame with no shadows
sbiasref[] = 0.0002f0    # too little to cover a PCF tap: acne on the grazing faces
sbiasref[] = 0.004f0     # back

aoamb[] = 0f0            # the same frame with no ambient occlusion
aoamb[] = 6f0            # far too much, which is the way to see where it acts
aoamb[] = 2.6f0          # back

aoradius[] = 1.5f0       # tight crevices only, rather than whole clusters
aoradius[] = 4.5f0       # back

bloomamt[] = 2.5f0       # more glow
bloomamt[] = 0.30f0      # back

threshold[] = 0.4f0      # what counts as bright
threshold[] = 1.6f0      # back

emitref[] = 25f0         # the sources themselves, much hotter
emitref[] = 9f0          # back

exposure[] = 2f0
exposure[] = 0.85f0      # back

fogref[] = 0.02f0        # thicker air
fogref[] = 0.0035f0      # back

suncolref[] = Vec4f(0.55, 0.72, 1.0, 2.5)   # a cold moon instead of a warm sun
suncolref[] = Vec4f(1.0, 0.55, 0.28, 6.5)   # back

sunref[] = Vec4f(normalize(Vec3f(-0.62, 0.045, 0.35))..., 0)   # lower, longer shadows
sunref[] = Vec4f(normalize(Vec3f(-0.62, 0.22, 0.35))..., 0)    # back

freeze_cull = true       # the frustum stops following the camera
freeze_cull = false

speed = 0f0              # hold still

M.timings(plan)          # per pass; a sum down the column is not the frame time

round(M.peakbytes(plan) / 2^20, digits = 1)   # what the transients cost together

begin # ── stop ──────────────────────────────────────────────────────────────
    stop[] = true
    wait(render)
    close(win)
end
