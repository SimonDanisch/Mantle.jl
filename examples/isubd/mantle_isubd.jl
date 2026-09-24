# Adaptive tessellation of curved FEM cells: the port of `reference.jl` onto Mantle.
# Every literal is Float32 (`1f0`), because Apple GPUs have no Float64 at all.

using Mantle, GeometryBasics, LinearAlgebra, FileIO, Colors
import Hikari
using KernelAbstractions: @kernel, @index, get_backend
using Hikari: monomials, eval_poly, surfacepoint, surfacenormal

@inline eval_position(cx, cy, cell, a, b) =
    (eval_poly(cx, cell, a, b), eval_poly(cy, cell, a, b))

function basetriangles(ncells::Integer = 2)
    rim = ((-1f0, -1f0), (1f0, -1f0), (1f0, 1f0), (-1f0, 1f0))
    corners = Float32[]
    cells = Int32[]
    for c in 1:ncells, i in 1:4
        a, b = rim[i], rim[mod1(i + 1, 4)]
        # (b, centre, a): the split edge (v1, v3) must be the element edge.
        append!(corners, (b[1], b[2], 0f0, 0f0, a[1], a[2]))
        push!(cells, Int32(c))
    end
    return corners, cells
end

@inline function basecorner(basecorners, t::Integer, j::Integer)
    o = (t - 1) * 6 + (j - 1) * 2
    return (basecorners[o + 1], basecorners[o + 2])
end

@inline key_base(k::UInt64) = Int(k >> 32)
@inline key_leb(k::UInt64) = k % UInt32
@inline make_key(b::Integer, leb::UInt32) = (UInt64(b) << 32) | UInt64(leb)
@inline root_key(b::Integer) = make_key(b, UInt32(1))
@inline key_depth(k::UInt64) = 31 - leading_zeros(key_leb(k))
@inline key_parent(k::UInt64) = make_key(key_base(k), key_leb(k) >> 1)
@inline key_child(k::UInt64, i) = make_key(key_base(k), key_leb(k) << 1 | UInt32(i))
@inline is_child0(k::UInt64) = iseven(key_leb(k))

@inline function key_corners(k::UInt64, basecorners)
    b = key_base(k)
    c1 = basecorner(basecorners, b, 1)
    c2 = basecorner(basecorners, b, 2)
    c3 = basecorner(basecorners, b, 3)
    w1 = (1f0, 0f0, 0f0); w2 = (0f0, 1f0, 0f0); w3 = (0f0, 0f0, 1f0)
    leb = key_leb(k)
    for i in (key_depth(k) - 1):-1:0
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

# Three flat tuples read at a literal index: a GPU cannot index a nested tuple at runtime.
@inline sample_a() = (0.5f0, 0.5f0, 0f0,   1f0/3f0)
@inline sample_b() = (0f0,   0.5f0, 0.5f0, 1f0/3f0)
@inline sample_c() = (0.5f0, 0f0,   0.5f0, 1f0/3f0)

@inline function excess_levels(k::UInt64, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol)
    cell = Int(basecell[key_base(k)])
    p1, p2, p3 = key_corners(k, basecorners)
    x1 = eval_position(cx, cy, cell, p1...); f1 = eval_poly(cf, cell, p1...)
    x2 = eval_position(cx, cy, cell, p2...); f2 = eval_poly(cf, cell, p2...)
    x3 = eval_position(cx, cy, cell, p3...); f3 = eval_poly(cf, cell, p3...)
    per = ntuple(Val(4)) do si
        a = sample_a()[si]; b = sample_b()[si]; c = sample_c()[si]
        q1 = a * p1[1] + b * p2[1] + c * p3[1]
        q2 = a * p1[2] + b * p2[2] + c * p3[2]
        ex, ey = eval_position(cx, cy, cell, q1, q2)
        lx = a * x1[1] + b * x2[1] + c * x3[1]
        ly = a * x1[2] + b * x2[2] + c * x3[2]
        return (sqrt((ex - lx)^2 + (ey - ly)^2),
                abs(eval_poly(cf, cell, q1, q2) - (a * f1 + b * f2 + c * f3)))
    end
    geo_err = maximum(map(first, per))
    fld_err = maximum(map(last, per))
    return max(log2(max(geo_err, 1f-16) / geo_tol), log2(max(fld_err, 1f-16) / fld_tol))
end

q2nodes() = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
             (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0))

function q2basis()
    nmono = Hikari.FEM_NMONO
    V = zeros(nmono, nmono)
    for (k, (a, b)) in enumerate(q2nodes())
        V[k, :] .= collect(monomials(a, b))
    end
    return inv(V)
end

function annulus(; nθ = 20, nr = 3, r0 = 0.40, r1 = 1.15, lobes = 3, amp = 0.46)
    Vinv = q2basis()
    nmono = Hikari.FEM_NMONO
    ncells = nθ * nr
    cx = zeros(Float32, nmono, ncells); cy = zeros(Float32, nmono, ncells)
    cf = zeros(Float32, nmono, ncells)
    xs = zeros(nmono); ys = zeros(nmono); fs = zeros(nmono)
    c = 0
    for j in 1:nr, i in 1:nθ
        c += 1
        for (k, (a, b)) in enumerate(q2nodes())
            u = (i - 1 + 0.5 * (a + 1)) / nθ
            v = (j - 1 + 0.5 * (b + 1)) / nr
            θ = -2π * u  # clockwise, so the normals point up
            r = r0 + (r1 - r0) * v
            xs[k] = r * cos(θ)
            ys[k] = r * sin(θ)
            fs[k] = 0.5 + amp * cos(lobes * θ) * v
        end
        cx[:, c] = Vinv * xs; cy[:, c] = Vinv * ys; cf[:, c] = Vinv * fs
    end
    return cx, cy, cf
end

@kernel function classify_kernel!(counts, @Const(keys), @Const(cx), @Const(cy), @Const(cf),
                                  @Const(basecorners), @Const(basecell),
                                  geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        d = key_depth(k)
        if d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0
            counts[i] = Int32(2)
        elseif d > 0 && excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0
            counts[i] = is_child0(k) ? Int32(1) : Int32(0)
        else
            counts[i] = Int32(1)
        end
    end
end

@kernel function scatter_kernel!(out, @Const(keys), @Const(counts), @Const(offsets),
                                 @Const(cx), @Const(cy), @Const(cf),
                                 @Const(basecorners), @Const(basecell),
                                 geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        n = counts[i]
        o = offsets[i]
        if n == Int32(2)
            out[o + 1] = key_child(k, 0)
            out[o + 2] = key_child(k, 1)
        elseif n == Int32(1)
            d = key_depth(k)
            merging = d > 0 &&
                      excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0 &&
                      !(d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0)
            out[o + 1] = merging ? key_parent(k) : k
        end
    end
end

function refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
    n = length(keys)
    counts = similar(keys, Int32, n)
    backend = get_backend(keys)
    # The coefficients' precision: a Float64 tolerance would promote the kernel to double.
    geo_tol = eltype(cf)(geo_tol)
    fld_tol = eltype(cf)(fld_tol)
    classify_kernel!(backend, 64)(counts, keys, cx, cy, cf, basecorners, basecell,
                                  geo_tol, fld_tol, max_depth; ndrange = n)
    offsets = accumulate(+, counts) .- counts
    total = Int(scalarat(offsets, n)) + Int(scalarat(counts, n))
    out = similar(keys, total)
    scatter_kernel!(backend, 64)(out, keys, counts, offsets, cx, cy, cf,
                                 basecorners, basecell,
                                 geo_tol, fld_tol, max_depth; ndrange = n)
    return out
end

scalarat(a::AbstractArray, i::Integer) =
    Mantle.isdevicearray(a) ? only(Array(view(a, i:i))) : a[i]

function refine(keys0, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth = 12)
    keys = keys0
    for _ in 1:(max_depth + 1)
        new = refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
        length(new) == length(keys) && Array(new) == Array(keys) && break
        keys = new
    end
    return keys
end

rootkeys(nbase::Integer = 8) = UInt64[root_key(b) for b in 1:nbase]

# Plain functions, not globals: these are read inside GPU stages.
@inline meshthreads() = 32
@inline strokecolor() = (0.06f0, 0.07f0, 0.10f0)
@inline envwidth() = 4096
@inline envheight() = 2048

strokehalfpx() = 0.55f0

sundirection(; azimuth = deg2rad(140.0), elevation = deg2rad(38.0)) =
    Vec3f(cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))
lightdirection() = normalize(sundirection())

function demoenv(; sun = sundirection(), intensity = 1.0f0, turbidity = 2.2f0,
                 groundtile = 0.34f0, sunangle = 2.5f0, sunradiance = 830.0f0)
    envlight, _ = Hikari.sunsky_to_envlight(direction = sun,
                                                   intensity = intensity,
                                                   turbidity = turbidity,
                                                   ground_enabled = false,
                                                   resolution = 1024)
    sq = copy(envlight.env_map.data)
    n = size(sq, 1)
    acc = Hikari.RGBSpectrum(0f0); cnt = 0
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        if abs(d[3]) < 0.02f0
            acc = Hikari.RGBSpectrum((acc.c .+ sq[j, i].c)...); cnt += 1
        end
    end
    hz = Hikari.RGBSpectrum((acc.c ./ max(cnt, 1))...)
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        d[3] >= 0f0 && continue
        t = (-d[3])^2
        f = 1f0 - 0.72f0 * t
        gx = d[1] / max(-d[3], 1f-3)
        gy = d[2] / max(-d[3], 1f-3)
        r = sqrt(gx * gx + gy * gy)
        fade = clamp(1f0 - r / 9f0, 0f0, 1f0)
        sx = sinpi(gx / groundtile); sy = sinpi(gy / groundtile)
        w = clamp(0.05f0 * (1f0 + 0.35f0 * r), 0.05f0, 1f0)
        tile = 0.55f0 + 0.45f0 * (0.5f0 * (1f0 + clamp(sx * sy / w, -1f0, 1f0)))
        g = 1f0 - fade * (1f0 - tile)
        sq[j, i] = Hikari.RGBSpectrum(hz.c[1]*f*g*0.96f0, hz.c[2]*f*g*0.94f0, hz.c[3]*f*g*0.88f0)
    end
    sd = normalize(sun)
    cosr = cos(deg2rad(sunangle))
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        c = d[1]*sd[1] + d[2]*sd[2] + d[3]*sd[3]
        c <= cosr && continue
        t = clamp((c - cosr) / ((1f0 - cosr) * 0.34f0), 0f0, 1f0)
        w = sunradiance * (t * t * (3f0 - 2f0 * t))
        sq[j, i] = Hikari.RGBSpectrum(sq[j, i].c[1] + w,
                                      sq[j, i].c[2] + w * 0.95f0,
                                      sq[j, i].c[3] + w * 0.86f0)
    end
    return sq
end

function demosky()
    sq = demoenv()
    eq = Hikari.equalarea_to_equirect(sq)
    return map(c -> Colors.RGB{Float32}(c.c[1], c.c[2], c.c[3]), eq)
end

function skyenv(; w = envwidth(), h = envheight(), gain = 1.0f0)
    img = demosky()
    H, W = size(img)
    out = Array{Float32}(undef, 3 * w * h)
    for j in 1:h, i in 1:w
        si = clamp(round(Int, (i - 0.5) / w * W + 0.5), 1, W)
        sj = clamp(round(Int, (j - 0.5) / h * H + 0.5), 1, H)
        c = img[sj, si]
        f(x) = (v = Float32(x); isfinite(v) ? gain * max(v, 0f0) : 0f0)
        k = 3 * ((j - 1) * w + (i - 1))
        out[k+1] = f(Colors.red(c)); out[k+2] = f(Colors.green(c)); out[k+3] = f(Colors.blue(c))
    end
    return out
end

@inline function sampleenv(env, dx::Float32, dy::Float32, dz::Float32)
    l = sqrt(dx*dx + dy*dy + dz*dz) + 1f-8
    x = dx / l; y = dy / l; z = dz / l
    u = 0.5f0 + atan(y, x) * 0.15915494f0        # 1/(2pi)
    v = acos(clamp(z, -1f0, 1f0)) * 0.31830987f0 # 1/pi, 0 at the zenith
    fx = clamp(u, 0f0, 1f0) * (envwidth() - 1) + 1f0
    fy = clamp(v, 0f0, 1f0) * (envheight() - 1) + 1f0
    i0 = clamp(unsafe_trunc(Int32, fx), Int32(1), Int32(envwidth() - 1))
    j0 = clamp(unsafe_trunc(Int32, fy), Int32(1), Int32(envheight() - 1))
    tx = fx - Float32(i0); ty = fy - Float32(j0)
    @inline function at(i, j)
        k = 3 * ((Int(j) - 1) * envwidth() + (Int(i) - 1))
        return (env[k+1], env[k+2], env[k+3])
    end
    a = at(i0, j0); b = at(i0 + Int32(1), j0)
    c = at(i0, j0 + Int32(1)); d = at(i0 + Int32(1), j0 + Int32(1))
    w00 = (1f0-tx)*(1f0-ty); w10 = tx*(1f0-ty); w01 = (1f0-tx)*ty; w11 = tx*ty
    return (a[1]*w00 + b[1]*w10 + c[1]*w01 + d[1]*w11,
            a[2]*w00 + b[2]*w10 + c[2]*w01 + d[2]*w11,
            a[3]*w00 + b[3]*w10 + c[3]*w01 + d[3]*w11)
end

@inline tonemap(x::Float32) = clamp((x / (1f0 + x))^(1f0/2.2f0), 0f0, 1f0)

@inline function toclip(x, y, z, mvp::Mat4f)
    p = mvp * Vec4f(Float32(x), Float32(y), Float32(z), 1.0f0)
    return p
end

@inline function shade(env, r::Float32, g::Float32, b::Float32,
                       n::Vec3f, viewdir::Vec3f, lightdir::Vec3f, exposure::Float32)
    sun_diffuse = 0.45f0
    sun_specular = 2.0f0
    sun_tint = (1.0f0, 0.965f0, 0.90f0)
    sky_ambient = 0.95f0
    ndl = max(n[1]*lightdir[1] + n[2]*lightdir[2] + n[3]*lightdir[3], 0f0)
    ndv = clamp(-(n[1]*viewdir[1] + n[2]*viewdir[2] + n[3]*viewdir[3]), 0f0, 1f0)

    sr, sg, sb = sampleenv(env, n[1], n[2], n[3])

    d = viewdir[1]*n[1] + viewdir[2]*n[2] + viewdir[3]*n[3]
    rx = viewdir[1] - 2f0*d*n[1]; ry = viewdir[2] - 2f0*d*n[2]; rz = viewdir[3] - 2f0*d*n[3]
    er, eg, eb = sampleenv(env, rx, ry, rz)
    fres = 0.03f0 + 0.42f0 * (1f0 - ndv)^5

    hx = lightdir[1] - viewdir[1]
    hy = lightdir[2] - viewdir[2]
    hz = lightdir[3] - viewdir[3]
    hl = sqrt(hx*hx + hy*hy + hz*hz) + 1f-8
    ndh = max((n[1]*hx + n[2]*hy + n[3]*hz) / hl, 0f0)
    ndh2 = ndh * ndh; ndh4 = ndh2 * ndh2; ndh8 = ndh4 * ndh4
    ndh16 = ndh8 * ndh8; ndh32 = ndh16 * ndh16
    spec = sun_specular * ndh32 * ndh32 * (ndl > 0f0 ? 1f0 : 0f0)

    sun = sun_diffuse * ndl
    lr = exposure * (r * (sun * sun_tint[1] + sky_ambient * sr) + fres * er + spec * sun_tint[1])
    lg = exposure * (g * (sun * sun_tint[2] + sky_ambient * sg) + fres * eg + spec * sun_tint[2])
    lb = exposure * (b * (sun * sun_tint[3] + sky_ambient * sb) + fres * eb + spec * sun_tint[3])
    return (tonemap(lr), tonemap(lg), tonemap(lb))
end
